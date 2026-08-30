# c10-vectors-wiring

Follow-up batch opened from c5's carry-over list and c9's `testVectors`
limitation: **make the vector subsystem reachable.**

c5 ported `Lucene99FlatVectorsFormat` and the whole HNSW stack and verified
them properly -- this port's searcher over Lucene's own graph reproduces
`KnnFloatVectorQuery` doc-for-doc, a graph built here is 4273/4273 nodes
identical to Lucene's, and real Lucene reads Rust-written vector files. But
nothing in `lucene-index` could *add* a vector field, so none of it was
reachable from `add_document`, and c9 had to record `testVectors` as
"expressible but with no producer".

Three things landed:

1. **`IndexWriter` indexes vector fields.** `set_vector_field` /
   `add_vector_field` / `add_document_with_vectors` / `set_hnsw_parameters`,
   with the flush writing real `PerFieldKnnVectorsFormat`-named
   `.vec`/`.vemf`/`.vem`/`.vex` and the `.fnm` attributes that make them
   findable. Verified end to end: a 3000-document index written by
   `IndexWriter` is opened by real `DirectoryReader`, searched with real
   `KnnFloatVectorQuery`/`KnnByteVectorQuery` against Lucene's own brute-force
   top-k, and passes real `CheckIndex`.
2. **The codec-level merge entry points c5 left open**:
   `Lucene99FlatVectorsWriter.mergeOneFlatVectorField`,
   `Lucene99HnswVectorsWriter.mergeOneField`, `IncrementalHnswGraphMerger`,
   `MergingHnswGraphBuilder`, `InitializedHnswGraphBuilder` and
   `UpdateGraphsUtils`. Not wired into `merge.rs` -- that file belongs to
   `c8-tv-chunking` this cycle; the wiring is an explicit handoff below.
3. **c5's `lucene-util` carry-over**: `SplittableRandom`, `TernaryLongHeap` and
   the `NumericUtils` float helpers moved out of `lucene-codecs/src/hnsw.rs`.

Java counterparts (Lucene 10.5.0 at `/home/tuong/work/lucene`):

- `index/IndexingChain.java` (`PerField.knnFieldVectorsWriter`,
  `initializeFieldInfo`), `document/{KnnFloatVectorField,KnnByteVectorField}.java`,
  `codecs/{KnnVectorsWriter,KnnFieldVectorsWriter}.java`,
  `codecs/perfield/PerFieldKnnVectorsFormat.java`
- `codecs/lucene99/{Lucene99FlatVectorsWriter,Lucene99HnswVectorsWriter}.java`
- `util/hnsw/{IncrementalHnswGraphMerger,HnswGraphMerger,MergingHnswGraphBuilder,
  InitializedHnswGraphBuilder,UpdateGraphsUtils,HnswGraphBuilder,HnswGraphSearcher,
  OnHeapHnswGraph}.java`
- `util/{TernaryLongHeap,NumericUtils}.java`, `java.util.SplittableRandom`

Totals: **45 findings** -- 5 CORRECTNESS (all fixed), 23 MISSING (19 fixed, 4
recorded with named blockers), 3 PERF (1 fixed, 2 measured), 13 INTENTIONAL
(one of which -- finding 36 -- is a fix for c5's recorded layering
carry-over). Finding 24 is a **correctness defect in Java itself**
(`InitializedHnswGraphBuilder.rebalanceGraph` draws from an unseeded RNG, so
Lucene's own merge is irreproducible when deletions are present); this port
diverges deliberately by seeding it.

---

## crates/lucene-index/src/index_writer.rs

Java counterparts: `index/IndexingChain.java`,
`document/{KnnFloatVectorField,KnnByteVectorField}.java`,
`codecs/KnnVectorsWriter.java`, `codecs/perfield/PerFieldKnnVectorsFormat.java`,
`codecs/lucene99/Lucene99HnswVectorsWriter.flush`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `set_vector_field` / `add_vector_field` | `IndexingChain.getOrAddPerField` + `initializeFieldInfo`'s vector half | ported, opt-in (finding 1); the dimension/encoding/similarity come from the `FieldInfo`, as they do in Java from the `IndexableFieldType` |
| `add_document_with_vectors` | `IndexingChain.processField`'s `fieldType.vectorDimension() != 0` branch -> `PerField.knnFieldVectorsWriter.addValue` | ported (finding 4); the per-document validation is `KnnFloatVectorField`'s constructor plus `IndexingChain.addField`'s duplicate-field check |
| `DocumentVector` / `VectorValue` | `KnnFloatVectorField` / `KnnByteVectorField` | one enum instead of two classes; the variant fixes the encoding |
| `set_hnsw_parameters` | `Lucene99HnswVectorsFormat(maxConn, beamWidth)` | ported, with Java's own bounds |
| `build_vectors_output` | `Lucene99HnswVectorsWriter.flush` (= `Lucene99FlatVectorsWriter.flush` + `writeField`/`writeGraph` on top) | ported, same order: flat store written and **reopened** first, graph built over the reopened values (finding 6) |
| `write_vector_files` | `PerFieldKnnVectorsFormat.FieldsWriter`'s suffixed file naming + `SegmentInfo.addFile` | ported (finding 3) |
| `fields_with_per_field_attributes` (vector half) | `PerFieldKnnVectorsFormat.getSuffix` + `FieldInfo.putAttribute` | ported, plus the zeroing rule (finding 2) |

Java with **no** Rust counterpart: `IndexingChain`'s incremental
`KnnFieldVectorsWriter.addValue` (this port buffers documents and inverts /
encodes at flush, which is b9's shape for every format); `writeSortingField`'s
caller (`IndexWriter` has no index-sorted flush at all -- finding 11);
`FLOAT16` (finding 17); `MergeState`-driven merge (handed off).

### Findings

1. **[MISSING -> fixed]** *`lucene-index` could not index a vector field.*
   c5's carry-over, and the reason c9 could not exercise its own
   `testVectors`/`testHnswGraphs`. `IndexWriter` now has
   `set_vector_field`/`add_vector_field` (a list, like `postings_fields`, so
   any number of vector fields land in one segment) and
   `add_document_with_vectors`. A flush writes the four files for every opted-in
   field that any pending document carried, honouring `should_create_graph`'s
   tiny-segment skip per field.
   *Tests*: `add_document_with_vectors_writes_a_readable_vector_segment`
   (dense FLOAT32 + sparse BYTE, every ordinal's value and doc id checked, a
   real graph for the big field and **no** graph for the sub-threshold one),
   `the_flushed_graph_finds_the_same_documents_an_exhaustive_scan_does`,
   `each_flushed_segment_gets_its_own_vector_files`,
   `set_hnsw_parameters_reaches_the_written_graph`, plus the whole
   `VerifyVectorSegment` case (below).

2. **[CORRECTNESS -> fixed]** *A `.fnm` claiming vectors the segment does not
   carry.* Java never reaches this state: `IndexingChain` sets
   `FieldInfo.vectorDimension` from the first document that actually carries
   the field, so the `.fnm` and the `.vemf` are written from one fact. This
   port takes its `FieldInfo` list from the caller at `open`, so a field
   declared with a dimension but given a value by no document would be written
   with `vector_dimension > 0` and no field entry in the `.vemf` and no
   `PerFieldKnnVectorsFormat` attribute. `FieldInfo.hasVectorValues()` is then
   true -- which is what `IncrementalHnswGraphMerger.addReader` and
   `CheckIndex` key off -- while `PerFieldKnnVectorsFormat` registers no reader,
   so the field reads back as vector-capable and yields nothing.
   **Measured, not assumed**: real Lucene raises *no error* for that
   combination (the negative control below opens and passes `CheckIndex`), which
   is exactly why it has to be got right here rather than left to fail loudly.
   `fields_with_per_field_attributes` now zeroes `vector_dimension` for every
   field this flush wrote no vectors for.
   *Tests*: `a_vector_field_no_document_carried_is_not_claimed_in_the_fnm`,
   `a_flush_with_no_vectors_writes_no_vector_files`, and the fixture's
   `never_written` field, asserted on by `VerifyVectorSegment`.

3. **[MISSING -> fixed]** *`PerFieldKnnVectorsFormat`'s naming and attributes.*
   Real Lucene writes `<segment>_Lucene99HnswVectorsFormat_0.{vec,vemf,vem,vex}`
   and records `PerFieldKnnVectorsFormat.format` / `.suffix` on each vector
   field's `.fnm` entry (confirmed against `fixtures/data/vectors_index`, which
   real Lucene wrote). Without both, `DirectoryReader` opens the segment, reports
   the field as having no vectors, and raises nothing -- the same silent shape c4
   found for postings and doc values. The four files are also added to
   `SegmentInfo.files`, without which `IndexFileDeleter`, `CheckIndex` and this
   port's own `checksum_verify` are blind to them.
   *Tests*: `the_segment_info_lists_the_four_vector_files`,
   `a_vector_field_no_document_carried_is_not_claimed_in_the_fnm` (checks the
   attributes are present on the written field and absent on the other), and a
   **negative control**: renaming the attribute key makes
   `VerifyVectorSegment` report all four fields as having no vectors.

4. **[MISSING -> fixed]** *No per-document validation.* Java validates in the
   `KnnFloatVectorField`/`KnnByteVectorField` constructor (dimension, and
   `VectorUtil.checkFinite`) and in `IndexingChain.addField`
   (`"VectorValuesField \"...\" appears more than once in this document"`).
   `add_document_with_vectors` rejects a wrong dimension, a `f32` value for a
   BYTE field or vice versa, a field that is not opted in, and the same field
   twice on one document -- all before buffering, so a rejected document cannot
   desynchronise the parallel buffers. `VectorUtil.checkFinite` stays where c5
   put it, in `write_flat_vectors`, because that is the last point that sees the
   whole field; a NaN therefore fails the `commit`, not the `add`.
   *Test*: `add_document_with_vectors_rejects_malformed_input` (one case each,
   plus the NaN-at-commit case and a `pending_doc_count() == 0` assertion).

5. **[INTENTIONAL]** *Vectors live in a buffer parallel to the document
   buffer, not inside `Document`.* `lucene_codecs::stored_fields::Document` is
   this port's **stored-fields** document: everything in it is serialized into
   `.fdt`. Java's `Document` is a list of `IndexableField`s of which
   `KnnFloatVectorField` is one *non-stored* kind, so putting a vector there
   would store every embedding in the stored-fields file, which Lucene does not
   do. `pending_vectors` is aligned 1:1 by index with `pending_docs`, the same
   convention c7's `pending_custom_freq_terms` established.
   *Test*: `the_vector_buffer_stays_aligned_with_the_document_buffer` drives
   every entry point that touches the buffer (`add_document`, `add_documents`,
   `add_document_with_vectors`, `rollback`) and then checks that the two
   surviving vectors landed on docs 1 and 4 -- a drift here does not fail
   loudly, it silently attaches every vector to the wrong document.

6. **[INTENTIONAL]** *The graph is built over the reopened `.vec`, not over the
   in-memory `Vec<f32>`.* Java's `Lucene99FlatVectorsWriter.flush` returns a
   scorer supplier over the bytes it just wrote, and
   `Lucene99HnswVectorsWriter` builds on that. Building over the in-memory
   values would be equivalent only as long as the two agree -- and an
   `alignOutput` or ordinal-assignment mistake is exactly the kind of thing that
   makes them not. Same reason `write_vectors_fixture.rs` already did it.

7. **[INTENTIONAL]** *A field with no values in this flush is skipped
   entirely*, rather than written with `count = 0`. Java reaches the same state
   from the other side (no `KnnFieldVectorsWriter` is created for a field no
   document carried), so its `.vemf` has no entry either.

8. **[PERF, measured]** *What vectors cost per document.* See Measurements: at
   the fixture's scale the flat store is free and the HNSW graph is essentially
   the whole cost.

### Verdict

Swept clean for the flush path. The subsystem c5 built is now reachable from
`add_document`, and the segments it produces are read, searched and
`CheckIndex`-ed by real Lucene. Open: the index-sorted flush (finding 11), and
`merge.rs` wiring (handed off).

---

## crates/lucene-index/src/indexing_chain.rs

Java counterpart: `index/IndexingChain.java`.

**No change, and that is the finding.**

9. **[INTENTIONAL]** *`IndexingChain`'s vector half does not belong in this
   module.* This port's `indexing_chain.rs` is only Java's tokenize-and-invert
   half (`TermsHashPerField`); it takes `(doc_id, field, text)` triples and an
   `Analyzer` and returns an inverted index. A vector is not analyzed, not
   tokenized and not inverted -- Java's `PerField.knnFieldVectorsWriter` sits
   beside `invertState`, not inside it, and `processField` routes a vector
   straight to it. Routing vectors through this module would mean widening its
   input type to carry values it never looks at. The vector path therefore lives
   where the other non-inverted formats' do, in `index_writer.rs`'s flush
   (`build_doc_values_output`, `build_norms_output`, now
   `build_vectors_output`).

### Verdict

Swept clean; no vector work belongs here.

---

## crates/lucene-index/src/segment_writer.rs

Java counterpart: `index/DocumentsWriterPerThread.flush`.

**No change.**

10. **[INTENTIONAL]** *Vector files are written by `index_writer.rs` after
    `flush_stored_only_segment_with_blocks` returns, and patch the `.si` in
    place.* That is the established pattern for every non-stored format here
    (`write_postings_files`, `write_doc_values_files`, `write_norms_files`,
    `write_term_vector_files`), and it is what lets a segment carrying postings
    *and* doc values *and* vectors end up with one `.si` listing all of them --
    whichever write ran first is read back and extended, not overwritten. An
    eleventh parameter on `flush_stored_only_segment` would have had to be
    threaded through every existing caller for nothing.

11. **[MISSING, recorded -- and a live cross-batch hazard, see below]** *The
    vectors index-sort write path has no reachable caller.* `Lucene99FlatVectorsWriter.writeSortingField` and
    `Lucene99HnswVectorsWriter.{writeSortingField,reconstructAndWriteGraph}`
    remap a field's ordinals through `Sorter.DocMap` when the segment is
    index-sorted. This module has `flush_sorted_stored_only_segment`, but
    **`IndexWriter::flush` never calls it** -- grep confirms the only references
    are doc comments in `merge.rs`/`segment_info.rs`/`index_writer.rs`. So a
    sorted flush cannot happen at all, with or without vectors, and porting the
    vector half would be porting a branch nothing can take. The prerequisite is
    an index-sorted `IndexWriter::flush`, which is a `b9`/`c3`-shaped task, not
    a vector one. Recorded rather than done.

    **This stopped being hypothetical during this batch.** A concurrent batch
    is adding `IndexWriter::set_index_sort` (`Error::EmptyIndexSort`,
    `IndexSortChangedMidBuffer`, `validate_index_sort_against_existing_segments`,
    and a `pending_sort_map: Option<(String, Vec<usize>)>` field). As of this
    writing `pending_sort_map` is **declared but never read** -- the sorted
    flush is not wired into `flush()` yet -- so nothing is broken today. But the
    moment it is wired, `build_vectors_output` becomes wrong in a way that is
    silent: it assigns ordinals by walking `pending_vectors` in **buffer order**
    and records `docs[i] = i`, so a flush that reorders the stored fields and
    the postings but not the vectors produces a segment in which **every vector
    is attached to the wrong document**. It decodes cleanly, passes `CheckIndex`
    (the vectors are individually well-formed and the ord->doc map is
    monotonic), and answers every KNN query with plausible, wrong documents.

    The fix is Java's, and it is small: `Lucene99FlatVectorsWriter.writeSortingField`
    remaps ordinals through `Sorter.DocMap` (`mapOldOrdToNewOrd`) before writing,
    and `Lucene99HnswVectorsWriter.reconstructAndWriteGraph` relabels the graph's
    arcs through the same map rather than rebuilding. Concretely, in
    `build_vectors_output`: iterate documents in **sorted** order, push
    `new_doc_id` rather than the buffer index, and build the `old_ord -> new_ord`
    permutation the graph writer needs. The graph half can reuse this batch's
    own machinery -- `HnswGraphBuilder::init_graph(scorer, beam_width, seed,
    &built_graph, &old_to_new_ord, count)` is exactly "copy this graph into a
    relabelled ordinal space", which is what `reconstructAndWriteGraph` does,
    and it is already tested arc-for-arc under a reversed ordinal map
    (`a_permuted_ordinal_map_relabels_every_arc`).

    Owner: whoever lands the sorted flush. Flagged here because the failure is
    silent and the two halves are being written in parallel.

### Verdict

Swept clean; the one open item (11) is blocked on a non-vector prerequisite.

---

## crates/lucene-codecs/src/vectors.rs

Java counterparts: `codecs/lucene99/Lucene99FlatVectorsWriter.java`
(`mergeOneFlatVectorField`, `writeMeta`, `alignOutput`),
`codecs/KnnVectorsWriter.MergedVectorValues`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `FlatVectorsWriter::{new,write_field,finish}` | `Lucene99FlatVectorsWriter.{ctor,writeField,finish}` | ported (finding 14); `write_flat_vectors` is now a thin wrapper for the flush path |
| `FlatVectorsWriter::merge_one_flat_vector_field` | `mergeOneFlatVectorField` + `MergedVectorValues.merge{Float,Byte}VectorValues` + `write{,Byte}VectorData` | ported (finding 12), with a bulk-copy improvement (finding 13) |
| `FlatVectorsWriter::{align_output,write_meta}` | `alignOutput`, `writeMeta` | identical, now shared by both entry points |
| `MergeSourceValues` / `FlatVectorMergeSource` / `MergedFlatVectorField` | `MergeState`'s per-field slice of `knnVectorsReaders` + `docMaps` | data-shaped, since this port has no `MergeState` (finding 12) |
| `{Float,Byte}VectorValues::raw_range` | (no counterpart -- Java re-encodes per vector) | finding 13 |
| `validate_docs` | `DefaultFieldWriter.addValue`'s ordering rules + `IndexedDISI.writeBitSet`'s precondition | factored out of `validate_field` so the merge path enforces them too (finding 15) |

Java with **no** Rust counterpart: `writeSortingField`/`writeSortedFloat32Vectors`/
`mapOldOrdToNewOrd`; `mergeFloat16VectorValues` and everything FLOAT16;
`getMergeInstance`/`finishMerge`/`updateIOContext`; `DocIDMerger`'s sorted-index
variant (the merged doc order here is source order, which is what a non-sorted
merge produces).

### Findings

12. **[MISSING -> fixed]** *`mergeOneFlatVectorField` was not ported.*
    `FlatVectorsWriter::merge_one_flat_vector_field` takes the sources' opened
    `{Float,Byte}VectorValues` and their doc maps and writes the merged field
    straight into the `.vec` being built. Two Java properties are kept. First,
    **no buffering**: the merged field is never materialized, so merging a 1M x
    768 field costs no heap beyond the output. Second, the merged ordinal space
    is defined *here and nowhere else* -- ordinals are assigned in source order
    and, within a source, in its own ordinal order -- which is exactly the
    ordering `hnsw_vectors::merge_one_field`'s ordinal maps assume.
    *Test*: `a_merge_produces_exactly_what_one_flush_of_the_same_documents_would`
    compares the merged `.vec` **and** `.vemf` **byte for byte** against a single
    flush of the same documents. That is the real contract -- a merge may not be
    observable in the output -- and it is strictly stronger than checking the
    values decode back. Also `a_merge_drops_deleted_documents_and_renumbers_the_rest`
    (a drop inside a run and one at the end, so both the flush-the-run and
    start-a-new-run branches fire).

13. **[PERF -> fixed]** *One `memcpy` per surviving run, not one re-encode per
    vector.* Java's `writeVectorData` copies each vector into a `ByteBuffer`,
    `asFloatBuffer().put(value)`, and writes it -- per vector. The on-disk
    representation of a FLOAT32 vector is little-endian `f32`, which is
    byte-for-byte what this port writes, so a run of consecutive surviving
    ordinals can be copied straight out of the source's mapped `.vec`. With no
    deletions -- the common case -- that is **one copy per source segment**
    instead of one per vector; with deletions it is one per unbroken run. No
    decode, no re-encode, no per-vector bounds check.

14. **[MISSING -> fixed]** *The writer had no incremental form.* Java's
    `Lucene99FlatVectorsWriter` is a stateful consumer with `writeField`,
    `mergeOneFlatVectorField` and `finish`; this port had one function taking
    every field at once, which a merge cannot use (a merged field is not a
    `FlatVectorsField`, and materializing one would undo finding 12).
    `FlatVectorsWriter` is that consumer. `write_flat_vectors` is retained as
    the flush-path convenience, so no existing caller changed.
    *Test*: `a_flushed_field_and_a_merged_field_share_one_file_pair`, which also
    asserts both fields' `alignOutput` padding survived the mixing.

15. **[MISSING -> fixed]** *The merge could emit a doc list its own reader
    accepts and `IndexedDISI` mis-encodes.* c5's finding 6 added ascending /
    in-range / non-negative validation to the flush path; a merge with a wrong
    doc map produces exactly that shape of bad list, from a different direction.
    `validate_docs` is now shared, and the merge additionally cross-checks the
    bytes it wrote against `count * dim * byteSize` -- which catches a source
    whose `size()` and `ord_to_doc` disagree before the `.vemf` records a length
    the `.vec` does not have.
    *Tests*: `merge_rejects_a_doc_map_that_is_not_ascending_or_is_short`
    (backwards map, duplicate target, short map),
    `merge_checks_every_source_against_the_declared_field` (encoding mismatch,
    dimension mismatch, non-positive dimension).

16. **[INTENTIONAL]** *`VectorUtil.checkFinite` is not re-run on merge.* A
    vector reaching the merge was checked when its source segment was written,
    and Java does not re-check either. Re-checking would mean decoding every
    FLOAT32 vector, which is exactly what finding 13 avoids.

17. **[MISSING, recorded]** *FLOAT16.* Genuinely not contained. Adding a third
    `VectorEncoding` variant ripples through `field_infos.rs`'s
    `from_byte`/`to_byte` (which currently *rejects* ordinal 2), every `match`
    on the enum in `lucene-search` and `lucene-ffi` -- both owned by other
    batches this cycle -- and needs a new `Gen*.java` fixture, an
    `OffHeapFloat16VectorValues` (IEEE half decode) and half-precision variants
    of all four similarity transforms. It is a batch, not a follow-on. Recorded
    with the blocker named.

### Verdict

Swept clean for the flat merge, and the merged bytes are proven identical to a
flush of the same documents. Open: FLOAT16 (17), the index-sort remap (11).

---

## crates/lucene-codecs/src/hnsw.rs

Java counterparts: `util/hnsw/{HnswGraphBuilder,InitializedHnswGraphBuilder,
MergingHnswGraphBuilder,UpdateGraphsUtils,OnHeapHnswGraph,HnswGraphSearcher}.java`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `HnswGraphView::max_node_id` | `HnswGraph.maxNodeId()` + `OnHeapHnswGraph.maxNodeId()`'s `noGrowth` branch | **added** (finding 18) |
| `OnHeapHnswGraph::with_size` | `new OnHeapHnswGraph(M, numNodes)` | ported, including the refusal to grow |
| `HnswGraphBuilder::with_graph` | the `HnswGraphBuilder(scorerSupplier, beamWidth, seed, hnsw)` ctor | ported; `M` comes from the graph, as in Java |
| `add_graph_node_with_entry_points` / `add_graph_node_internal(_, eps0)` | `addGraphNode(node, eps0)` / `addGraphNodeInternal`'s `eps0` branch | ported (finding 19), incl. `beamCandidates0` and its narrower `NeighborArray` |
| `add_diverse_neighbors(_, _, _, is_link_repair)` / `select_and_link_diverse(..., is_link_repair)` | the same two with `isLinkRepair` | ported (finding 20) |
| `compute_join_set` / `join_set_coverage` / `encode_gain` / `decode_gain*` | `UpdateGraphsUtils.{computeJoinSet,coverage,encode,decodeValue1,decodeValue2}` | ported (finding 21) |
| `HnswGraphBuilder::init_graph` / `initialize_from_graph` / `copy_graph_structure` / `fix_disconnected_nodes` / `add_connections` / `rebalance_graph` | `InitializedHnswGraphBuilder.{initGraph,initializeFromGraph,copyGraphStructure,fixDisconnectedNodes,addConnections,rebalanceGraph}` | ported (finding 22), one RNG divergence (24) |
| `merge_graphs` | `MergingHnswGraphBuilder.fromGraphs(...).build(maxOrd)` | ported as one function (findings 23, 28) |
| `HnswGraphBuilder::update_graph` | `MergingHnswGraphBuilder.updateGraph` | ported; entry-point set order differs (finding 26) |

Java with **no** Rust counterpart: `ConcurrentHnswMerger` /
`HnswConcurrentMergeBuilder` (concurrent merge -- this port builds
single-threaded, c5 finding 11); `HnswGraphMerger`'s interface (one
implementation here); `InfoStream` progress printing; `setAbortCheck`
(`MergePolicy.MergeAbortedException` has no equivalent yet).

### Findings

18. **[CORRECTNESS -> fixed]** *The searcher sized its `visited` bitset from
    `size()`, where Java uses `maxNodeId() + 1`.* `HnswGraphSearcher` has a
    static `getGraphSize(graph) { return graph.maxNodeId() + 1; }`, and both
    `findBestEntryPoint` and `searchLevel` size `visited` from it. c5 used
    `graph.size()`. For a *finished* graph the two are equal, which is why every
    c5 test passed -- but they differ precisely while a graph is being built
    **out of order**, which is what merging does: `InitializedHnswGraphBuilder`
    copies a source segment's nodes into their new ordinals before the rest of
    the merged ordinal space exists, so `size()` counts what has been added and
    is not a bound on what a search may visit. `visited.set(node)` with
    `node >= size()` is an out-of-bounds index -- a panic on the first merge
    that searches a partly-built graph, which is every merge with more than one
    source graph. `max_node_id` is now on `HnswGraphView` (defaulting to
    `size() - 1`, Java's own default) and `OnHeapHnswGraph` overrides it with
    Java's `noGrowth` rule.
    *Test*: `a_partly_built_graph_reports_max_node_id_from_its_declared_size`,
    and every merge test exercises the path.

19. **[MISSING -> fixed]** *`addGraphNode(node, eps0)` and `beamCandidates0`.*
    The merge's whole saving is that it seeds the level-0 search with entry
    points it already knows, and searches with a **narrower** beam
    (`min(beamWidth / 2, M * 3)`) because of it. Both are ported, including
    Java's `new NeighborArray(max(candidates.k(), M + 1), false)` reading `k()`
    off whichever collector was used -- a detail that silently changes the
    scratch array's capacity if missed.

20. **[MISSING -> fixed]** *`isLinkRepair`.* Two behaviours hang off it, and
    both matter only on the merge path. `updateNeighbor` scans for an existing
    `node -> nbr` arc and skips it, without which a repaired node gets a
    **duplicate arc** in the serialized `.vex`; and `selectAndLinkDiverse` uses
    `addOutOfOrder` rather than `addInOrder`, because during repair the array
    already holds surviving neighbours whose scores are unrelated to the new
    candidates', so `addInOrder`'s "each new entry is worse than the last"
    precondition does not hold (it is a `debug_assert` here, i.e. a panic in a
    debug build).
    *Test*: `assert_well_formed` in the merge tests checks no duplicate arcs, no
    self arcs, no out-of-range arcs, and the per-level budget, on every merged
    graph.

21. **[MISSING -> fixed]** *`UpdateGraphsUtils.computeJoinSet`.* The set of
    level-0 nodes that between them cover the small graph; those go into the big
    graph the ordinary way and every other node is inserted with entry points
    derived from them. Ported including `coverage`'s clamp to the node's own
    degree, which is what stops a degenerate degree-1 graph from putting every
    leaf in the join set (Java's comment calls this out explicitly).
    *Tests*: `the_join_set_is_a_small_cover_of_the_graph` (proper subset, and
    every node outside it has a neighbour inside it -- the property the merge
    actually depends on), `join_set_coverage_is_clamped_to_the_degree`,
    `the_gain_encoding_round_trips_and_orders_largest_first`.

22. **[MISSING -> fixed]** *`InitializedHnswGraphBuilder`.* All three phases:
    the structural copy with ordinal remapping (dropped neighbours simply not
    copied, `addOutOfOrder(_, NaN)` as the not-yet-scored marker), the
    disconnected-node repair with both of Java's thresholds
    (`DISCONNECTED_NODE_FACTOR = 0.85` for a big single-merge loss,
    `CUMULATIVE_DEGREE_FLOOR_FACTOR = 0.5` for slow decay across many merges),
    and the level rebalance. **With no deletions the whole thing is a pure
    structural copy** -- no search, no scoring, no RNG draw -- which is the
    property the incremental merge exists for.
    *Tests*: `merging_one_undeleted_graph_reproduces_it_arc_for_arc` (exact
    equality of every level's every node's neighbour list, in order -- the
    assertion that actually discriminates, per c5's Tier-2 finding that recall
    does not), `a_permuted_ordinal_map_relabels_every_arc` (a reversed ordinal
    map, which catches a copy that gets the structure right and forgets to
    remap -- invisible under an identity map),
    `merging_a_graph_with_deletions_leaves_no_dangling_arcs`.

23. **[MISSING -> fixed]** *`MergingHnswGraphBuilder`.* `merge_graphs` copies
    `graphs[0]` into the merged ordinal space, folds each later graph in via
    `update_graph`, and finally inserts any ordinal no source graph covered.

24. **[CORRECTNESS in Java, INTENTIONAL divergence here]** *Java's merge is not
    reproducible when the deletions branch runs.*
    `InitializedHnswGraphBuilder.rebalanceGraph` draws from
    `new SplittableRandom()` -- **unseeded** -- so two Lucene merges of
    identical inputs can produce different graphs. Everything else on the merge
    path is seeded (`HnswGraphBuilder.randSeed = 42`). This port seeds the
    rebalance RNG, which makes the merged graph a function of its inputs and
    therefore testable at all; nothing about the format or the search depends on
    *which* nodes get promoted, only on the distribution. Recorded here because
    it also settles a question this batch had to answer: an **arc-for-arc
    differential against Java's merged graph is not achievable** even in the
    no-deletions case, because `MergingHnswGraphBuilder.updateGraph` hands
    `IntHashSet.toArray()` to the search and hppc's iteration order is a hash
    order (finding 26). The evidence used instead is (a) exact arc-for-arc
    equality against the *source* graph for the pure-copy case, (b) real Lucene
    reading and searching a merge-written segment, and (c) the cost measurement.

25. **[INTENTIONAL]** *`computeJoinSet`'s `counts` widened from `short` to
    `i32`.* A node's cover count is bounded only by its in-degree, so Java's
    `short[]` can in principle wrap negative on a large graph and make an
    already-covered node look uncovered. Widening cannot change the answer for
    any input Java gets right, and removes an input class where it does not.

26. **[INTENTIONAL]** *Entry-point set order.* Java collects `updateGraph`'s
    entry points into an `IntHashSet` and passes `toArray()` to the search --
    hppc's hash order. This port keeps insertion order: deterministic, which
    hppc's is not across implementations, and the set is only a *seed* for a
    beam search, so its order can only break score ties.

27. **[INTENTIONAL]** *`beamCandidates0`'s size is clamped to at least 1.*
    Java writes `min(beamWidth / 2, M * 3)`, which is **0** for
    `beamWidth == 1` -- a zero-capacity collector whose `popNode` would then be
    called. Unreachable in Lucene only because nothing configures
    `beamWidth = 1`. A zero-capacity heap is a panic, not a behaviour, so it is
    clamped rather than reproduced.

28. **[INTENTIONAL]** *`merge_graphs` is one function, not a builder struct.*
    Java's `MergingHnswGraphBuilder` captures the source graphs and the ordinal
    maps at construction and reads them back in `build`; in Rust that is a
    struct borrowing the inputs and owning the output, i.e. a lifetime knot for
    no gain. The RNG is shared across the copy and fold phases instead of being
    reset between them, which is equivalent: the copy phase draws from it only
    in the deletions branch, and that branch has its own RNG in both languages.
    `rebalance_graph`'s `tryPromoteNewEntryNode` is likewise guarded rather than
    asserted (Java asserts `level > expectOldLevel`, which this port would turn
    into a debug-build panic on an input Java would silently accept in
    production).

29. **[MISSING -> fixed]** *A non-injective ordinal map indexed out of bounds.*
    `update_graph` reads a node's neighbours in the *target* graph, which is
    safe only because every referenced ordinal was inserted earlier. An ordinal
    map that maps two source ordinals onto one merged ordinal breaks that and
    the index panics; it is now an `InvalidGraphParameter` naming the ordinal.

### Verdict

Swept clean for merge-time construction, with one real CORRECTNESS fix (18)
that was latent for the flush path and fatal for the merge path. Open:
`ConcurrentHnswMerger` (single-threaded by design, c5 finding 11).

---

## crates/lucene-codecs/src/hnsw_vectors.rs

Java counterparts: `codecs/lucene99/Lucene99HnswVectorsWriter.{mergeOneField,
buildAndWriteGraph,createGraphMerger}`,
`util/hnsw/IncrementalHnswGraphMerger.java`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `merge_one_field` | `mergeOneField`'s graph half + `buildAndWriteGraph` + `createGraphMerger` + `IncrementalHnswGraphMerger.merge` | ported (finding 30) |
| the `addReader` loop inside it | `IncrementalHnswGraphMerger.addReader` | identical decisions, incl. `DELETE_PCT_THRESHOLD = 40` and the asymmetric size comparison (finding 32) |
| `count_live_vectors` | `countLiveVectors` | derived from the doc map (finding 31) |
| `new_ord_mapping` | `getNewOrdMapping` | identical, incl. the first-match-wins loop and the `initializedNodes` marking |
| the `createBuilder` ordering | `createBuilder`'s `addFirst` / `sort by graphSize desc` | identical |
| `GraphMergeSource` | `MergeState`'s per-source `knnVectorsReaders[i]` / `docMaps[i]` / `liveDocs[i]` | data-shaped |

Java with **no** Rust counterpart: `ensureFlatReaderOpen` (the caller reopens,
as the flush path already does); `QuantizedVectorsReader`'s branch
(quantization is out of scope); `ConcurrentHnswMerger`;
`mergeState.intraMergeTaskExecutor`.

### Findings

30. **[MISSING -> fixed]** *`mergeOneField` was not ported* (c5 finding 21).
    `merge_one_field` picks the largest usable source graph, copies it into the
    merged ordinal space and folds the rest in, or rebuilds from scratch when no
    source graph qualifies -- and returns `None` when `shouldCreateGraph` says
    the merged segment is too small for a graph, exactly as the flush path does.
    *Tests*: `merging_reuses_the_largest_graph_instead_of_rebuilding`,
    `merging_rebuilds_when_no_source_has_a_graph` (asserted **arc for arc**
    against `HnswGraphBuilder::build`, since a rebuild is exactly that),
    `merging_writes_no_graph_below_the_threshold`,
    `a_heavily_deleted_graph_is_not_used_as_the_base` (also arc-for-arc against
    a rebuild), `merge_rejects_a_source_whose_graph_and_vectors_disagree`, and
    `a_merged_segment_round_trips_through_both_readers` (the whole thing: two
    segments' four files in, one segment's out, reopened and searched at recall
    >= 0.9 against the exhaustive answer over the merged file).

31. **[INTENTIONAL]** *`liveDocs` is derived from the doc map.* Java passes
    both `MergeState.docMaps[i]` and `MergeState.liveDocs[i]` to `addReader`,
    and `countLiveVectors` uses the second. But "live" is exactly "the doc map
    maps this document somewhere", so taking both would be taking two inputs
    that can disagree. One input, derived.

32. **[INTENTIONAL]** *Java's asymmetric base-graph comparison is kept.*
    `addReader` compares the candidate's **live** count against the incumbent's
    **total** size (`candidateVectorCount > largestGraphReader.graphSize`), not
    against its live count. It only ever picks a base graph, and at equal size a
    graph with deletions is the worse base, which is what the asymmetry
    expresses -- so it is ported as written rather than "fixed".

33. **[PERF, measured]** *The merge costs a fraction of a rebuild.* Merging a
    1200-vector graph with a 120-vector one: **11,124 similarity computations
    against a rebuild's 122,473, 11.0x fewer**, counted with an instrumented
    scorer. Asserted (at a 3x floor) in
    `merging_reuses_the_largest_graph_instead_of_rebuilding`, so the merge
    cannot silently regress into a rebuild -- which is exactly the failure mode
    that would pass every structural and recall check.

34. **[MISSING -> fixed]** *Caller-input validation.* A source whose graph size
    disagrees with its vector count, or whose doc map has no entry for a doc its
    vectors name, would index out of bounds; an `M` outside `1..=512` would
    **panic** inside `OnHeapHnswGraph::with_size` before any builder saw it
    (this entry point allocates the merged graph itself, unlike
    `HnswGraphBuilder::new`, which returned an error). All of them are
    `InvalidGraphParameter` now, checked before anything is allocated.
    *Tests*: `merge_rejects_a_source_whose_graph_and_vectors_disagree`,
    `merge_rejects_out_of_range_graph_parameters`.

35. **[MISSING, recorded]** *`ConcurrentHnswMerger`.* c5 finding 11's scope
    decision (single-threaded construction) unchanged; the merge inherits it.

### Verdict

Swept clean. The merge entry points are complete and tested at the codec level,
and real Lucene reads and searches a merge-produced segment. Open: the
`merge.rs` wiring, handed off below.

---

## crates/lucene-util/src/{splittable_random,ternary_long_heap,numeric_utils}.rs (new)

Java counterparts: `java.util.SplittableRandom`,
`org.apache.lucene.util.TernaryLongHeap`, `org.apache.lucene.util.NumericUtils`.

36. **[INTENTIONAL -> fixed]** *c5 finding 28: three types that are not codec
    concerns lived in `lucene-codecs/src/hnsw.rs`.* `SplittableRandom` is a
    **JDK** class; `TernaryLongHeap` and the `NumericUtils` sortable-float
    helpers are `org.apache.lucene.util`. None is HNSW-specific in Java, and
    `util/hnsw` has no dependency on `codecs.lucene99` at all where this port's
    `hnsw.rs` did. All three now live in `lucene-util`, one module per Java
    concept per the **architecture** skill's "no `util`/`misc` dumping ground"
    rule, with `lucene-codecs` importing them downward. `TernaryLongHeap`'s
    methods became `pub` (they were private to `hnsw.rs`); nothing else changed.
    Their tests moved with them and were widened, since a `lucene-util` module
    is now the owner of its own boundary behaviour: 100% line coverage on all
    three, including `TernaryLongHeap`'s two panics and `get()`'s raw
    heap-array order (the property `NeighborQueue.nodes()` depends on and that a
    "cleaner" `BinaryHeap` would silently change).

37. **[MISSING, recorded]** *The `NumericUtils` long/double halves are still
    duplicated.* `lucene-index/src/segment_info.rs` has its own
    `float_to_sortable_int`/`double_to_sortable_long`, and
    `lucene-search/src/facets.rs` its own `doubleToSortableLong`/
    `sortableDoubleBits`. Both should point at `lucene_util::numeric_utils`, and
    the `*ToSortableBytes` pair used by `points.rs` and `points_query.rs`
    belongs there too. Not done here: neither file is this batch's, and
    `lucene-search` belongs to `c12-search-features-2` this cycle. Recorded.

### Verdict

Swept clean; the layering inversion c5 recorded is closed.

---

## Measurements

### What vectors cost per document

`benchmarks/rust-runner`'s `index-bench` gained a `LUCENE_RUST_VECTOR_DIM` knob
that gives every document a FLOAT32 `KnnFloatVectorField` of that dimension.
Same corpus, same machine, one core, `--release`, `M = 16`, `beamWidth = 100`,
default 16 MB RAM buffer.

**Below `HNSW_GRAPH_THRESHOLD` (600 documents, so no graph is built) --
this isolates the flat store:**

Five interleaved pairs:

| | µs/doc |
|---|---|
| no vectors | 23.9 / 21.7 / 20.0 / 20.6 / 20.2 (median **20.6**) |
| 128-dim vector per document | 22.0 / 20.3 / 20.9 / 22.4 / 24.4 (median **22.0**) |

The sign alternates (the vector arm is *faster* in two of five pairs):
**the flat store is free at this scale.** Writing `dim * 4` bytes per document
into a `Vec<u8>` plus an `IndexedDISI` bitset does not register against the
~21 µs a document already costs. Note this arm's baseline lands right on
c3/c7's 20-21 µs/doc, which is the reassurance that the 50k arm's higher
baseline is load, not a regression.

**Above the threshold (50 000 documents, so every flushed segment builds a
graph):**

Interleaved A/B, medians of 3-5 alternating pairs each (interleaved because
other batches were building on the same machine, and a block of runs at one
setting would absorb whatever else was happening):

| vector dimension | µs/doc (median) | range | vs vector-free |
|---|---|---|---|
| none | **25.8** | 24.7-30.6 | — |
| 16 | **85.5** | 80.2-97.9 | +59.7 |
| 64 | **142.9** | 136.6-157.1 | +117.1 |
| 128 | **203.8** | 194.3-203.9 | +178.0 |

So **the HNSW graph is essentially the entire cost**, and it scales with the
dimension because the dimension is what a similarity computation costs.

Two honest caveats. First, the vector-free baseline reads 25.8 µs/doc where
c3/c7 measured 20-21 µs/doc; the same binary ranged 24.7-42 µs/doc across the
session depending on what else was compiling, so the baseline carries roughly
that much noise and the *deltas* are the trustworthy part. Second,
the corpus is **uniform random** vectors -- the ANN worst case, which c5
measured at 2.4x more work per query than clustered data of the same shape; the
same penalty applies at build time, and it is why 128 dimensions costs ~178
µs/doc here against the ~104 µs/doc c5 measured for a 50k x 128 graph over
clustered data. A real embedding field is clustered.

### What the merge saves

Merging a 1200-vector graph with a 120-vector one, counted with an instrumented
scorer (`merging_reuses_the_largest_graph_instead_of_rebuilding`):

| | similarity computations |
|---|---|
| `merge_one_field` (incremental) | **11,124** |
| `HnswGraphBuilder::build` (rebuild) | 122,473 |

**11.0x fewer.** The test asserts a 3x floor, so a regression to a rebuild --
which would pass every structural check and every recall check -- fails.

---

## Verification

**Rust writes, real Lucene reads and searches** -- two new/extended cases,
`scripts/verify-write-path.sh` **17/17 -> 18/18** (a concurrent batch has since added a 19th case; the script is green at 19/19):

- `crates/lucene-index/examples/write_vector_segment_fixture.rs` ->
  `fixtures/src/VerifyVectorSegment.java` (**new**). A 3000-document index
  written by the real `IndexWriter` with four vector fields (dense FLOAT32
  EUCLIDEAN; sparse FLOAT32 COSINE; BYTE DOT_PRODUCT on the first 1500; a
  5-document MAXIMUM_INNER_PRODUCT field below the graph threshold) plus
  postings, norms and stored fields, and a fifth vector field declared in the
  `FieldInfo` list that no document ever carries. Java opens it with
  `DirectoryReader` and checks: every ordinal's components (an order-sensitive
  hash over raw bits, against values the Rust side derived from its *generator*,
  not read back out of the files it wrote) and every ordinal's doc id; the
  declared-but-never-written field is absent and records dimension 0; for each
  field and each of 12 queries, **Lucene's own brute-force top-k over the
  vectors it just read** against what `KnnFloatVectorQuery`/
  `KnnByteVectorQuery` return over the Rust-built graph; then postings, stored
  fields, and `CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS` (which runs Lucene's
  own `testVectors` and `testHnswGraphs`).
  Result: recall@10 **0.9083 / 0.9750 / 0.9750**, and the graphless field
  **exactly** matches brute force (asserted as set equality, not recall --
  Lucene searches it exhaustively, so anything less is a flat-store bug, not a
  graph one).

  Two **negative controls**, run by hand: renaming the
  `PerFieldKnnVectorsFormat.format` attribute key makes all four fields report
  `getFloatVectorValues returned null`; removing the `vector_dimension` zeroing
  makes the `never_written` field report dimension 6. Both segments still open,
  and the second still passes `CheckIndex` -- which is the whole reason the
  verifier asserts on the fields directly instead of trusting `CheckIndex`.

- `crates/lucene-codecs/examples/write_vectors_fixture.rs` ->
  `fixtures/src/VerifyVectors.java` (**extended**). A fifth field,
  `merged_f32`, is produced by the **merge** entry points rather than the flush
  ones: two sub-segments are written and reopened as real `.vec`/`.vemf` pairs,
  each gets its own graph, and the two are folded together by
  `FlatVectorsWriter::merge_one_flat_vector_field` +
  `hnsw_vectors::merge_one_field`. Everything Java checks for the other four
  fields it checks for this one -- the value hash, every ordinal's doc id, the
  graph's level count / entry node / max conn / per-level arc hash, and a real
  `TopKnnCollector` search. Result: **recall@10 0.9750**. The example also
  asserts, before Java sees anything, that the merged field's vectors are
  ordinal-for-ordinal what a flush of the same documents would have produced.

**This port's own `CheckIndex` over a writer-produced vector segment.** c9
ported `testVectors`/`testHnswGraphs` against Java-written fixtures only,
because nothing here could write a vector segment.
`our_own_check_index_passes_over_a_writer_produced_vector_segment` now runs
`check_index::check_directory` over a 900-document segment this writer
produced, and asserts it saw 2 vector fields and 1350 vector values -- so a
check that silently skipped the vectors would fail too.

**Unit tests.** 11 new in `index_writer.rs`, 6 in `vectors.rs`, 15 in
`hnsw.rs`, 7 in `hnsw_vectors.rs`, and full test modules for the three new
`lucene-util` files. Four of the `hnsw.rs` ones exist to reach branches the
merge path has but the happy path does not: `init_graph` on its own,
`with_graph`'s bounds (an `OnHeapHnswGraph` can legally be built past
`MAXIMUM_MAX_CONN`, so `with_graph` has to check what `new` checks before the
graph exists), a malformed graph view naming an ordinal past its own `size()`,
and -- the interesting one --
`deleting_the_upper_level_nodes_makes_the_merge_rebalance`, which had to
construct its deletions deliberately: a *uniform* deletion thins every level in
proportion and never triggers `rebalanceGraph` at all, so the test drops
exactly the nodes above level 0. The load-bearing ones are the exact-equality assertions, per c5's
Tier-2 conclusion that recall does not discriminate:

- the merged `.vec`/`.vemf` are **byte-identical** to a flush of the same
  documents;
- merging one undeleted graph reproduces it **arc for arc**, and under a
  reversed ordinal map every arc is exactly the relabelled original;
- a rebuild (no usable source graph, or a >40%-deleted one) is **arc for arc**
  `HnswGraphBuilder::build`;
- `assert_well_formed` rejects a duplicate arc, a self arc, an out-of-range
  arc, a level-0 set that is not every ordinal, and an over-budget node -- on
  every merged graph in the suite.

**Gates**: `cargo fmt --all`, `cargo clippy -p lucene-index -p lucene-codecs
-p lucene-util --all-targets -- -D warnings`, `cargo test -p lucene-index -p
lucene-codecs -p lucene-util`, and `scripts/verify-write-path.sh` (19/19, of which
this batch added one) -- all green. Coverage (`cargo llvm-cov --summary-only`, lines): `index_writer.rs`
98.29%, `hnsw.rs` 96.75%, `hnsw_vectors.rs` 95.60%, `vectors.rs` 96.94%, and
`lucene-util`'s three new files 100%.

One measurement note worth recording, because it wasted an hour: with other
batches running `cargo test` against the same target directory, `cargo
llvm-cov` silently mixes stale and fresh `.profraw` data and reports figures
tens of points low (`hnsw.rs` came back as 64.89% and then 69.62% before a run
with its own `CARGO_TARGET_DIR` gave 96.75%). A coverage number taken while
anything else is building the workspace is not a number.

### Tier 2 semantic review

Run on the diff (`quality-reviewer`). **No gating findings** -- the review
walked `computeJoinSet`, `copyGraphStructure`, `rebalanceGraph`,
`MergingHnswGraphBuilder.build`, `updateGraph`, `addReader`/`createBuilder`/
`getNewOrdMapping`, `TernaryLongHeap` and `mergeOneFlatVectorField` against the
Java line by line and found them faithful, and confirmed the buffer invariant
and the `maxNodeId` claim independently. It also traced Java's
`tryPromoteNewEntryNode` assertion and showed the guarded Rust form is exactly
equivalent rather than a behaviour change, and checked
`FieldInfo.checkConsistency` to confirm a zeroed `vector_dimension` with a
non-default encoding/similarity is legal rather than merely tolerated.

Nine advisories, **all acted on**:

38. **[CORRECTNESS -> fixed]** *`copy_graph_structure` bounds-checked the node
    ordinal against `new_ord_map` but indexed the **neighbour** ordinal
    unchecked* -- a short map panics there instead of returning the
    `InvalidGraphParameter` finding 29 went out of its way to produce. Not
    reachable through `merge_one_field` (which validates
    `ord_to_doc.len() == graph.size()`), but reachable through the public
    `merge_graphs`/`init_graph`. The map's length is now checked once against
    `initializer.size()`, which covers node *and* neighbour ids. The review also
    caught that `merge_graphs_rejects_mismatched_inputs`' short-map case only
    took the error path *by accident of which ordinal the level assignment put
    on top* -- so the test now pins the length rejection deterministically, and
    a second case proves a length-valid map is accepted.

39. **[CORRECTNESS -> fixed]** *A corrupt `.vex` could panic the merge.*
    `OffHeapHnswGraph::neighbors_into` bounds arc counts at `2 * M` on **every**
    level (as Java's `currentNeighborsBuffer` does), but an upper level's
    `NeighborArray` holds only `M + 1` -- so a checksum-valid `.vex` with an
    over-full upper-level node reached `add_out_of_order`'s `assert!` (a panic
    in release, not a debug assertion). `copy_graph_structure` now returns
    `CorruptMeta`. Java throws `IllegalStateException` for the same input, so
    this is a panic-vs-error difference, not a fidelity one.
    *Test*: `a_source_node_with_more_arcs_than_the_level_allows_is_rejected`,
    against a hand-built `HnswGraphView` that reports `M + 1` arcs on level 1.

40. **[CORRECTNESS -> fixed]** *`rebalance_graph` drew from
    `DEFAULT_RAND_SEED`, not the builder's own seed*, so finding 24's whole
    point -- "the merged graph is a function of its inputs" -- held for the
    inputs but not for the caller's `seed`. The builder now keeps its seed and
    the rebalance uses it.

41. **[MISSING -> fixed]** *A tautological assertion.* `assert_well_formed`'s
    "a node on level N is also on every level below" check drew the node from
    the very list it then searched, so it could never fail -- and it runs on
    every merged graph in the suite, reading as coverage that was not there. It
    now compares against level `N - 1`'s node list.

42. **[MISSING -> fixed]** *The flat merge left the writer half-written on
    error.* `merge_one_flat_vector_field` emits alignment padding (and possibly
    whole runs of copied vectors) before it can fail on a source mismatch, so a
    caller that recovered would carry those bytes into the next field. It now
    rolls `data` back to its pre-call length on every error path, which
    `merge_checks_every_source_against_the_declared_field` asserts before going
    on to write the field successfully through the same writer.

43. **[MISSING -> fixed]** *Setter timing was undocumented.* The opt-ins are
    read at **flush**, so `set_vector_field(None)` with vectors already buffered
    discards them silently, and `set_hnsw_parameters` mid-buffer re-shapes the
    graph the current buffer is about to produce (in Java both are fixed on the
    `FieldType`/codec before any document exists). Same timing every other
    opt-in here has, so it is documented on all three rather than guarded.

44. **[MISSING -> fixed]** *This report's own totals were wrong* (31 claimed,
    37 present) and `docs/parity.md`'s HNSW row still listed `SplittableRandom`
    inside `lucene-codecs/src/hnsw.rs` and mapped `util/TernaryLongHeap` there,
    where finding 36 had just moved both. `hnsw.rs`'s module doc claimed to port
    `TernaryLongHeap` too. All three corrected; the parity row now names the
    three `lucene-util` modules and the merge builders.
    The review's suggestion is worth recording as a project item: **an `xtask`
    that resolves every `path::Item` named in `parity.md`'s "Rust counterpart"
    column and fails when one does not** would have caught this mechanically,
    and stale paths are the recurring failure mode whenever a type changes
    crates.

45. **[MISSING -> fixed]** *`VerifyVectorSegment` hard-coded the graph
    threshold's crossing point* (`count > 650`). Correct today, but a change to
    `HNSW_GRAPH_THRESHOLD` or `expectedVisitedNodes` would silently downgrade
    the *exact*-match assertion on the graphless field into the weaker recall
    one. It now derives the predicate from
    `Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD` and Lucene's own formula.

The review also confirmed the flush path's
`should_create_graph(threshold, final_count)` is equivalent to Java's
*incremental* `shouldCreateGraph(threshold, node + 1)` plus
`replayBufferedVectors`, because `n - 100 ln n` is monotone above its crossing
and the replay inserts nodes in the same order -- so the RNG stream, and
therefore the graph, is identical. That equivalence was assumed by c5 and is
now checked.

---

## Handoff: wiring the vector merge into `merge.rs`

`crates/lucene-index/src/merge.rs` belongs to `c8-tv-chunking` this cycle, so
the wiring below is **not** done. The codec entry points are complete, tested
and real-Lucene-verified; what is missing is the caller. Precisely:

For each field of the merged `FieldInfos` with `vector_dimension > 0`:

1. **Open the sources.** Per source segment, open
   `<seg>_Lucene99HnswVectorsFormat_0.vemf` + `.vec` with
   `vectors::FlatVectorsReader::open(meta, data, &segment_id, &suffix)` where
   `suffix = index_writer::per_field_codec_suffix("Lucene99HnswVectorsFormat")`,
   and `.vem` + `.vex` with `hnsw_vectors::HnswVectorsReader::open` when the
   segment has them (a segment may legitimately have the flat pair and a
   `numLevels = 0` graph). A source whose `.fnm` gives the field
   `vector_dimension == 0` contributes nothing and must be skipped entirely --
   the same rule `Lucene99HnswVectorsWriter.buildAndWriteGraph` applies with
   `hasVectorValues(mergeState.fieldInfos[i], fieldInfo.name)`.

2. **Merge the flat store.** One
   `vectors::FlatVectorsWriter::new(merged_max_doc, &new_segment_id, &suffix)`
   for the whole segment; per field, one
   `merge_one_flat_vector_field(&MergedFlatVectorField { field_number,
   encoding, similarity, dimension, sources })` where `sources` is, **in merge
   order**, `FlatVectorMergeSource { values: MergeSourceValues::{Float32,Byte}(...),
   doc_map }`. `doc_map` is `merge.rs`'s existing per-source old-doc -> new-doc
   map with `-1` for a dropped document, indexed by the source's own doc id and
   at least `source_max_doc` long. Then `finish()` -> `(vec, vemf)`, written
   under `index_writer::per_field_segment(name, "Lucene99HnswVectorsFormat")`.
   The merged ordinal space is defined by this call; step 3 depends on it.

3. **Merge the graphs.** Reopen the `.vec`/`.vemf` just written. Per field,
   build `merged_ord_to_doc: Vec<i32>` from the reopened values, and call
   `hnsw_vectors::merge_one_field(merged_values.ord_scorer(), m, beam_width,
   hnsw::DEFAULT_RAND_SEED, &merged_ord_to_doc, &sources)` with
   `GraphMergeSource { graph: Option<&OffHeapHnswGraph>, ord_to_doc, doc_map }`
   **in the same order as step 2**, where `ord_to_doc` is the source's own
   ordinal -> its own doc id (materialize it from the source's
   `FloatVectorValues::ord_to_doc`) and `doc_map` is the same map as step 2.
   `m`/`beam_width` are the writer's configured values; note they are used only
   for a from-scratch rebuild, since a reused base graph supplies its own
   `maxConn` (Java does the same).

4. **Write the graph files.** `hnsw_vectors::write_hnsw_vectors` with one
   `HnswVectorsField` per field, `graph: result.as_ref()` (so `None` becomes
   `numLevels = 0`), `count = merged_values.size()`, and the same
   encoding/similarity/dimension. Write `.vem`/`.vex` under the suffixed
   segment name.

5. **Record them.** Add all four names to `SegmentInfo.files`, and stamp
   `PerFieldKnnVectorsFormat.format` / `.suffix` on each merged field's `.fnm`
   entry -- and **zero `vector_dimension` on every field that ended up with no
   vectors**, exactly as `index_writer::fields_with_per_field_attributes` does
   (finding 2; real Lucene raises no error for the inconsistent combination).

Two things that do **not** apply: there is no bulk-copy fast path for vectors
(Lucene has none either -- `mergeOneFlatVectorField` always re-writes the data
file, because the merged ordinal space is new), and `MatchingReaders` is
irrelevant for the same reason.

An end-to-end verifier for this is one line in `scripts/verify-write-path.sh`
once `write_merged_segment_fixture.rs` grows a vector field:
`VerifyMergedSegment` already opens the merged index with `DirectoryReader` and
runs `CheckIndex`, so it needs only the same `KnnFloatVectorQuery` assertions
`VerifyVectorSegment` now has.

---

## Carry-over

- [x] **`merge.rs` wiring** -- the handoff above. **Closed by
      `c22-sorted-merge`** (`merge::merge_vectors`), following the five steps
      with one addition the recipe could not have known about: an
      index-sorted merge interleaves its sources, so
      `merge_one_flat_vector_field` had to start assigning merged ordinals in
      **merged-document** order (`DocIDMerger.of(subs, needsIndexSort)`)
      rather than source order -- source order produces a descending step that
      `.vemf`'s `IndexedDISI` cannot encode. The per-run `memcpy` is kept.
- [ ] **`FLOAT16`** (finding 17). Blocked on a third `VectorEncoding` variant,
      which touches `field_infos.rs`, `lucene-search` and `lucene-ffi`, and on a
      new `Gen*.java` fixture. A batch of its own.
- [x] **Index-sorted flush** (finding 11) -- **closed by `c17-index-sort`**,
      and the hazard flagged here was discharged rather than hit. That batch
      does not remap vector ordinals at all: it permutes the *document buffer*
      (including `pending_vectors`) once, before any format is built, so
      `build_vectors_output` assigns ordinals in sorted-doc order by
      construction and no per-format remap exists to forget. c17 confirmed the
      hazard was real by negative control -- dropping the `pending_vectors`
      permutation attaches every vector to the wrong document and **passes
      `CheckIndex`** -- so `VerifySortedSegment` asserts every vector
      component per doc id. Finding 11's prescription ("call
      `flush_sorted_stored_only_segment` from `IndexWriter::flush`") turned
      out to be wrong: that function sorts the stored fields only, so every
      other format would still address the original doc ids. See
      `docs/sweep/m2/c17-index-sort.md` findings 5 and 16.
- [ ] **`NumericUtils` deduplication** (finding 37): point
      `lucene-index/src/segment_info.rs` and `lucene-search/src/facets.rs` at
      `lucene_util::numeric_utils`, and move the `*ToSortableBytes` pair there
      too. Owners: whoever holds those files.
- [ ] **`ConcurrentHnswMerger`** (finding 35) and `SparseFixedBitSet` for the
      searcher's visited set above ~1M vectors (c5 finding 20) -- both carried
      forward unchanged.
- [ ] **`FieldInfos` cross-validation of dimension/similarity at read time**
      (c5 finding 16) -- still open on the read side. The *write* side is now
      doubly covered: `VerifyVectors` reads a Rust-written `.fnm` through real
      Lucene, and `VerifyVectorSegment` opens a Rust-written segment through
      `DirectoryReader`, so Lucene's own version of the check runs against our
      bytes in both shapes.
