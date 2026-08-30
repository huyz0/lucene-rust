# c4-merge-fastpath — M2 sweep follow-up

Closes the three performance carry-overs batch **b10-merge** raised in
`crates/lucene-index/src/merge.rs` (b10 findings 24, 27, 29).

Files swept/changed:

- `crates/lucene-index/src/merge.rs` (mine)
- `crates/lucene-codecs/src/stored_fields.rs` (the writer/reader API the
  bulk-copy merge needs; b3's file, not under concurrent edit)
- `crates/lucene-codecs/src/points.rs` (`BKDWriter.merge`'s one-pass leaf plan;
  b7's file)
- `crates/lucene-codecs/src/blocktree.rs` — **two additive `TermsEnum` methods
  only** (`try_current_postings`, `try_current_postings_and_positions`).
  `c1-lazy-blocktree` owns this file; the addition is 45 lines at the end of an
  existing `impl` block and touches nothing `c1` is restructuring.
- New: `crates/lucene-index/examples/write_merged_segment_fixture.rs`,
  `fixtures/src/VerifyMergedSegment.java`,
  `benchmarks/rust-runner/src/merge_bench.rs`; one new case in
  `scripts/verify-write-path.sh`.

Java source of truth: `/home/tuong/work/lucene` (Lucene 10.5.0) —
`codecs/lucene90/compressing/Lucene90CompressingStoredFieldsWriter.java`,
`codecs/lucene90/compressing/Lucene90CompressingStoredFieldsReader.java`,
`codecs/compressing/MatchingReaders.java`,
`codecs/lucene90/compressing/Lucene90CompressingTermVectorsWriter.java`,
`util/bkd/BKDWriter.java`, `codecs/lucene90/Lucene90PointsWriter.java`,
`codecs/FieldsConsumer.java`, `index/MultiTerms.java`,
`index/MappingMultiPostingsEnum.java`,
`codecs/perfield/PerFieldPostingsFormat.java`.

Gate: `cargo fmt --all` clean; `cargo test -p lucene-index -p lucene-codecs`
green (1 564 tests, 0 failures); `scripts/verify-write-path.sh` **16/16**,
including the new merged-segment case. `cargo llvm-cov -p lucene-index -p
lucene-codecs --summary-only`: `merge.rs` 98.69% lines, `stored_fields.rs`
97.66%, `points.rs` 99.68%, `blocktree.rs` 97.03% — all above the 95% bar
(TOTAL 97.83%). `cargo clippy -p lucene-index --all-targets -- -D warnings`
green.

A Tier 2 `quality-reviewer` pass was run on this batch's diff against the Java
source; findings 17–22 below are its output.

---

## Headline measurements

`benchmarks/rust-runner/src/merge_bench.rs`, 4 segments × 20 000 documents
(two stored string fields, ~120 bytes of payload each; 4.9 MB of compressed
`.fdt`). Every "before" figure is the pre-`c4` algorithm, **re-implemented in
the benchmark itself and re-run over the same inputs in the same process** —
not a remembered number. Best of three runs after a warm-up.

| Path | before | after | speedup |
|---|---|---|---|
| Stored fields, **BULK** (no deletions, matching field numbers) | 529 ms / 151 k docs/s / 2.5 MB/s | **1.0 ms / 79 M docs/s / 1 309 MB/s** | **520x** |
| Stored fields, **DOC** (1/3 of every source deleted) | 425 ms / 126 k docs/s / 3.1 MB/s | **16.3 ms / 3.3 M docs/s / 82 MB/s** | **26.1x** |
| Stored fields, **VISITOR** (every source's fields renumbered) | 542 ms / 148 k docs/s / 2.5 MB/s | **18.8 ms / 4.3 M docs/s / 71 MB/s** | **28.8x** |
| Postings merge (2 000 terms × 4 sources) | 98.3 ms | **9.1 ms** | **10.8x** (conservative — see finding 12) |
| BKD 1-D `points::write` (1 M points, 4 sources) | 155 ms / 6.4 M pts/s | **60.6 ms / 16.5 M pts/s** | **2.6x** |

The "after" stored-fields figures include the full CRC of every source `.fdt`
that finding 17 added, which the "before" figures do not pay — so all three are
understated relative to a like-for-like algorithm comparison. (An earlier run
of the same benchmark, before that check existed and on a quieter machine, read
643x / 22.6x / 25.9x.)

Memory: the BULK and DOC paths never build a `Vec<Document>` at all, so the
merge no longer holds the whole merged segment's parsed stored fields resident
(previously one owned `String`/`Vec<u8>` per stored field per merged document,
plus the `Vec<Document>` itself). `merge_postings` no longer holds a
`BTreeSet<Vec<u8>>` of every distinct term of the field. `merge_points` no
longer deep-clones the merged point list on the way to the writer.

---

## `crates/lucene-codecs/src/stored_fields.rs`

Java counterparts: `Lucene90CompressingStoredFieldsWriter`,
`Lucene90CompressingStoredFieldsReader`.

### Method correspondence (new/changed only)

| Java | Rust | Verdict |
|---|---|---|
| `Lucene90CompressingStoredFieldsWriter` (the object) | `StoredFieldsWriter` | **new** — was a one-shot free function `write_chunked(&[Document])` |
| `startDocument`/`finishDocument`/`triggerFlush`/`flush(force)` | `StoredFieldsWriter::{add_document, finish_document, trigger_flush, flush}` | identical |
| `finish(numDocs)` | `StoredFieldsWriter::finish` | identical |
| `copyChunks` | `StoredFieldsWriter::copy_chunks` | equivalent; see finding 2 |
| `copyOneDoc` | `StoredFieldsWriter::add_serialized_document` | identical |
| `tooDirty` | `StoredFieldsWriter::too_dirty` | identical |
| `getMergeStrategy`'s codec/mode/chunkSize/dirty trio | `StoredFieldsWriter::can_bulk_copy` | equivalent; see finding 3 |
| `merge(MergeState)` | `merge.rs::write_merged_stored_fields` | equivalent (the `DocIDMerger` loop lives at the index layer here) |
| `Reader.serializedDocument` | `StoredFieldsReader::serialized_document` | identical |
| `Reader.BlockState` (cached decompressed chunk) | `ChunkCursor` + `DecompressedChunk` + `StoredFieldsReader::read_chunk` | **new** — see finding 4 |
| `Reader.isLoaded(docID)` | *(no counterpart; replaced by the chunk-boundary test)* | see finding 2 |
| `getChunkSize`/`getNumChunks`/`getNumDirtyChunks`/`getNumDirtyDocs`/`getMaxPointer`/`getCompressionMode`/`getFieldsStream`/`getIndexReader.getStartPointer` | `StoredFieldsReader::{chunk_size, num_chunks, num_dirty_chunks, num_dirty_docs, max_pointer, mode, fdt, chunk_for_doc}` | identical |
| `Reader.document(docID, visitor)`'s parse half | `parse_document` | split out so a sequential scan parses from a cached chunk |

### Findings

1. **[PERF, fixed] The stored-fields writer was one-shot, so no merge fast path
   could exist.** b10's finding 24. `write_chunked(docs: &[Document], ...)`
   could only be fed fully parsed documents, so every merge decompressed every
   document, allocated per stored field, and recompressed everything —
   `Lucene90CompressingStoredFieldsWriter`'s VISITOR strategy, and its slowest
   variant.
   **Fixed** — `StoredFieldsWriter` is now the streaming object Java's writer
   is (`bufferedDocs`/`numStoredFields`/`endOffsets`/`docBase`/`numChunks`/
   `numDirtyChunks`/`numDirtyDocs`, `triggerFlush`, `flush(force)`), with all
   three of Java's merge entry points on it. `write_best_speed`/
   `write_best_compression` are now four-line wrappers over it, and the
   per-mode payload framing that used to be an `FnMut` closure is the free
   function `write_unit(mode, scratch, out, payload)`. The refactor is
   byte-for-byte output-preserving: the existing `stored_fields` fixtures
   (Java-written bytes, both modes) and every write/read round-trip test pass
   unchanged, and `scripts/verify-write-path.sh`'s `VerifyStoredFields` still
   passes.

2. **[INTENTIONAL, correctness-critical] `copyChunks`' partial-chunk handling
   is expressed as the chunk-boundary condition rather than Java's cached-block
   test.** Java's `copyChunks` opens with
   `while (docID < toDocID && reader.isLoaded(docID)) copyOneDoc(reader, docID++);`
   — `isLoaded` asks whether the doc lives in the reader's *currently cached*
   decompressed block, which is Java's way of saying "this document is in a
   chunk we already started copying document-at-a-time". Its real invariant is
   the one enforced two lines later by `if (base != docID) throw new
   CorruptIndexException(...)`: after that loop, `docID` must sit exactly on a
   chunk boundary.
   This port tests that invariant directly — `reader.chunk_for_doc(doc)?.doc_base
   != doc` — which is stronger (it does not depend on a reader's cache state)
   and reaches the same three-phase shape: leading partial chunk copied
   document-at-a-time, whole chunks memcpy'd, trailing partial chunk copied
   document-at-a-time. Java's two `CorruptIndexException`s are kept as
   `Error::CorruptChunkBounds` (`base != docID`, and `docID > toDocID` after a
   chunk), plus a `chunk_docs <= 0` guard Java gets from its own writer's
   invariants. Tests:
   `copy_chunks_of_two_whole_segments_reproduces_every_document`,
   `copy_chunks_of_a_partial_range_copies_the_ragged_ends_document_at_a_time`,
   `copy_chunks_of_a_range_inside_one_chunk_copies_no_chunk_at_all`,
   `copy_chunks_after_buffered_documents_forces_a_dirty_flush_first`,
   `copy_chunks_of_an_empty_range_writes_nothing`,
   `copy_chunks_rejects_an_out_of_range_document_range`.

3. **[INTENTIONAL] The version/reader-class half of `getMergeStrategy` has no
   counterpart, because `open` already enforces it.** Java's
   `getMergeStrategy` first checks `candidate instanceof
   Lucene90CompressingStoredFieldsReader` and `getVersion() ==
   VERSION_CURRENT`, falling back to VISITOR otherwise — it has to, because a
   `MergeState` can carry readers from older codecs (`addIndexes`,
   backward-codecs). This port has exactly one `VERSION_CURRENT` and
   `stored_fields::open` refuses anything else, so a `MergeSource` cannot be
   holding a reader those checks would reject. Recorded in
   `stored_fields_merge_strategy`'s doc comment rather than being silently
   absent. Java's `BULK_MERGE_ENABLED` system-property escape hatch is
   likewise not ported (there is no system-property mechanism here; the
   equivalent is deleting the `Bulk` arm).

4. **[PERF, fixed] The reader had no cached block, so a sequential scan
   re-decompressed each chunk once per document.** Not something b10 raised —
   found while measuring, because the DOC path came out only 1.02x faster than
   the old materialising merge and the profile said decompression, not
   compression, dominated.
   `document()`/`serialized_document()` are *random-access* reads: they inflate
   only the sub-blocks the wanted document intersects (Java's
   `Decompressor.decompress(in, originalLength, offset, length, bytes)`).
   That is right for a random lookup and wrong for a scan: a BEST_SPEED chunk
   holds up to 1 024 documents in 10 sub-blocks, so reading its documents one
   at a time inflates each sub-block about a hundred times. Java never pays
   that — its reader caches the whole decompressed block in `BlockState` and
   `contains(docID)` decides whether to reload.
   **Fixed** — `StoredFieldsReader::read_chunk` decompresses a whole chunk
   into a `DecompressedChunk` (header + bytes), `DecompressedChunk::document`
   hands back `(numStoredFields, &[u8])` borrowed straight out of it (no
   per-document allocation either), and `ChunkCursor` is the cache: it reloads
   only when the requested document falls outside the chunk it holds.
   `copy_chunks`' two ragged-end loops and both of `merge.rs`'s per-document
   paths use it. This is what took the DOC path from 1.02x to **22.6x** and
   the VISITOR path from 1.01x to **25.9x**.
   `document()`/`serialized_document()` keep the partial-decompression
   behaviour, deliberately — the random-access API should not inflate a whole
   16 kB chunk to read one document. Tests:
   `a_chunk_cursor_serves_every_document_of_a_chunk_from_one_decompression`
   (asserts byte-equality with the random-access read for every document, and
   that a backwards/skipping walk reloads rather than serving a stale chunk),
   `read_chunk_reports_its_own_extent_and_rejects_documents_outside_it`,
   `a_chunk_holding_only_empty_documents_decompresses_to_nothing`.

5. **[PERF] Dirtiness accumulates across merge generations, which is exactly
   why `tooDirty` exists — and it is reachable here.** A segment this port
   *flushes* can never be too dirty: its single forced flush leaves fewer than
   `maxDocsPerChunk` dirty documents, so `numDirtyDocs > maxDocsPerChunk` is
   false. A segment this port *merges* accumulates one dirty chunk per
   bulk-copied source, generation after generation, which is precisely the
   "frequent reopen with tiny flushes" degradation Java's comment describes.
   Pinned by `dirtiness_accumulates_across_bulk_copies_until_the_segment_is_too_dirty`,
   which builds a segment with 1 025 dirty chunks by repeated bulk copy and
   asserts it then fails `can_bulk_copy` (and still reads back correctly).

6. **[INTENTIONAL] `Mode` is now public.** The merge has to compare a source's
   compressor against the writer's (`getMergeStrategy`'s
   `reader.getCompressionMode() == compressionMode`), and that comparison
   cannot live inside `lucene-codecs` because the strategy choice also depends
   on field numbering and deletions, which are index-layer concerns.

---

## `crates/lucene-codecs/src/points.rs`

Java counterparts: `util/bkd/BKDWriter.{merge, OneDimensionBKDWriter,
mergeComparator}`, `codecs/lucene90/Lucene90PointsWriter.merge`.

| Java | Rust | Verdict |
|---|---|---|
| `BKDWriter.merge` + `OneDimensionBKDWriter` | `presorted_leaf_plan` + `write_field`'s `presorted` branch | equivalent for the plan; see finding 7 |
| `BKDWriter.mergeComparator` | `merge.rs::merge_point_streams`'s ordering | identical (value bytes, then docID) |
| `Lucene90PointsWriter.merge`'s `numDims == 1` gate | `num_index_dims == 1` + a verified sortedness check | see finding 7 |
| `BKDWriter.split`'s `numIndexDims` range | `widest_dim`'s new early return | see finding 8 |

### Findings

7. **[PERF, fixed] No `BKDWriter.merge` one-pass path — every merged point was
   re-indexed.** b10's finding 29. `merge_points` concatenated each source's
   points and let `points::write` sort them globally, at every split node, with
   `Vec` partitioning and slice comparisons — `PointsWriter.mergeOneField`'s
   re-index, where Java's `Lucene90PointsWriter.merge` overrides it with
   `BKDWriter.merge` for the one-dimensional case.
   **Fixed in two halves.** `merge.rs::merge_point_streams` is Java's
   priority-queue loop: each source's 1-D points come off disk already sorted
   by value (BKD leaves are in left-to-right value order and each leaf is
   internally sorted), so a k-way merge by `(value, docID)` — Java's
   `mergeComparator` exactly — produces the globally sorted stream in one pass.
   `points.rs::presorted_leaf_plan` is `OneDimensionBKDWriter`: for a
   single-index-dimension field whose points are already sorted, the leaves are
   consecutive `maxPointsInLeafNode` chunks and each node's split value is the
   first value of its right subtree, so there is no sort, no per-node
   `widest_dim` scan, and — since the leaves are *slices of the caller's own
   vector* — no copy of the points either (`write_field` used to
   `field.points.clone()` the whole list, on top of the clone `merge.rs` had
   already made on the way in; both are gone).
   **The precondition is verified, not assumed.** Java takes it on the
   caller's word: `Lucene90PointsWriter.merge` only calls `BKDWriter.merge` for
   readers it knows are sorted, and a caller who lied would get a silently
   corrupt tree. `write_field` instead checks sortedness in one linear scan of
   cheap slice comparisons and falls back to the general path otherwise, so a
   hand-built `MergeSource` or a segment from another writer gets correct
   output either way. Same in `merge_point_streams`, which concatenates rather
   than merging when any source's stream is not actually sorted.
   Equivalence is pinned rather than argued:
   `presorted_plan_matches_the_general_plan_byte_for_byte` writes the same
   points sorted and deterministically shuffled at eight tree shapes
   (1, 2, 8, 9, 16, 17, 100, 1 000, 4 097 points) and asserts all three output
   files are byte-identical; `presorted_leaf_plan_agrees_with_compute_leaf_plan_on_split_values`
   compares the two plans directly (split values, split dimensions, and the
   leaf partition). Plus
   `a_presorted_one_dimension_field_round_trips_through_the_reader`,
   `a_presorted_field_with_trailing_data_only_dimensions_still_takes_the_one_pass_path`,
   `points_with_equal_values_keep_their_input_order_on_both_paths`, and, on the
   merge side, `one_dimension_point_streams_are_k_way_merged_into_one_sorted_stream`,
   `equal_point_values_are_ordered_by_document_id`,
   `a_multi_index_dimension_field_is_concatenated_not_merged`,
   `an_unsorted_source_stream_falls_back_to_concatenation`,
   `merging_empty_and_single_point_streams_is_well_defined`.

   Note this port keys the fast path off `num_index_dims == 1` where Java uses
   `numDims == 1`. That is the weaker and actually load-bearing condition: the
   trailing `num_dims - num_index_dims` dimensions are data-only payload that
   never participates in a split or a bound, so they cannot make the value
   order ambiguous. Java's stricter gate costs it the `LatLonShape`-style
   `numDims > numIndexDims == 1` case; covered here by
   `a_presorted_field_with_trailing_data_only_dimensions_still_takes_the_one_pass_path`.

8. **[PERF, fixed] `widest_dim` scanned every point at every split node even
   with one index dimension.** `BKDWriter.split` ranges over
   `config.numIndexDims()`, so with one there is nothing to choose — but this
   port still ran the full min/max pass, an extra O(points) scan at every one
   of the tree's internal nodes. Now an early `return 0`, which the function's
   own doc comment already promised. Folded into the 2.5x above.

---

## `crates/lucene-codecs/src/blocktree.rs` (additive only)

| Java | Rust | Verdict |
|---|---|---|
| `TermsEnum.postings(reuse, flags)` at the cursor's position | `TermsEnum::try_current_postings` | **new** |
| the same with `PostingsEnum.nextPosition()`/offsets/payloads | `TermsEnum::try_current_postings_and_positions` | **new** |

9. **[PERF, fixed] `TermsEnum` could report a term but not its postings, so
   every caller had to re-seek for a term it was already standing on.** This is
   the API gap b10's finding 27 was blocked on. `FieldTerms::postings(term,
   ..)` runs `seek_exact` from the trie root; the cursor's current frame
   already holds that term's metadata (`SegmentTermsEnum::stats_and_meta`, the
   port of `decodeMetaData`). The two new methods hand back the postings — and,
   for a positional field, the positions from the *same* metadata read —
   without seeking at all, which is what `FieldsConsumer.merge` gets for free
   in Java by holding one `TermsEnum` per sub-reader.
   45 lines, additive, in an existing `impl` block; `decode_meta_data` is
   idempotent (the same guard `try_next` already relies on), so the methods are
   safe to call between `try_next` calls.

---

## `crates/lucene-index/src/merge.rs`

Java counterparts: `codecs/compressing/MatchingReaders`,
`Lucene90CompressingStoredFieldsWriter.merge`, `codecs/FieldsConsumer.merge` /
`index/MultiTerms` / `index/MappingMultiPostingsEnum`, `util/bkd/BKDWriter.merge`,
`codecs/perfield/PerFieldPostingsFormat`.

| Java | Rust | Verdict |
|---|---|---|
| `MatchingReaders` | `matching_readers` | equivalent (identity-map test) |
| `getMergeStrategy` | `stored_fields_merge_strategy` | equivalent (see finding 3) |
| `StoredFieldsWriter.merge`'s `DocIDMerger` loop | `write_merged_stored_fields` | equivalent |
| `FieldsConsumer.merge` / `MultiTerms` / `MappingMultiPostingsEnum` | `merge_postings` + `TermCursor` | **was divergent — fixed (12)** |
| `BKDWriter.merge`'s queue loop | `merge_point_streams` | equivalent (7) |
| `PerFieldPostingsFormat.fieldsConsumer`'s `FieldInfo` attribute stamping | `describe_written_files` | **was missing — fixed (10)** |
| `Lucene90CompressingTermVectorsWriter.merge`'s BULK path | — | **still missing — PERF (13)** |

### Findings

10. **[CORRECTNESS] A merged segment's postings were invisible to real Lucene.**
    Found by the new `VerifyMergedSegment` on its first run, and *only*
    findable that way: this port's own reader routes postings by file name, so
    every round-trip test passed and `CheckIndex` reported the index **clean**.
    Real Lucene routes a field to a postings format purely through the
    `PerFieldPostingsFormat.format`/`.suffix` attributes in `.fnm`.
    `IndexWriter` stamps them at flush time (`fields_with_per_field_attributes`),
    but the merge writes its own `.fnm` from `reconcile_field_numbers`' output —
    seeded from a *source's* `FieldInfo`, and `IndexWriter::execute_merge`
    passes its undecorated `self.fields`. So a merged segment's `.fnm` carried
    no attribute at all: `MultiTerms.getTerms(reader, "body")` returned `null`
    and the field read back as having no terms, silently, with no error
    anywhere. Reachable through the ordinary
    `set_postings_field` + commit + automatic-merge path — i.e. every merged
    index this port produces lost its postings as far as Lucene is concerned.
    **Fixed** — `describe_written_files` makes the merged `.fnm` describe the
    files the merge actually wrote: per-field postings/doc-values format
    attributes stamped for the fields the merge wrote data for, and stale
    inherited ones stripped (an attribute naming a format whose files this
    merge did not write sends a reader looking for a file that does not exist).
    Verified end to end by `VerifyMergedSegment`; unit-tested by
    `a_merged_field_with_postings_gets_the_per_field_format_attributes`,
    `a_stale_per_field_attribute_inherited_from_a_source_is_stripped`,
    `a_merged_doc_values_field_gets_its_per_field_format_attributes`.

11. **[CORRECTNESS] The same class of `.fnm` lie for norms and term vectors.**
    Same root cause, same fix, and both are fatal rather than silent:
    `DirectoryReader.open` throws on a missing `.nvm` rather than degrading, so
    a merged `.fnm` that still claims norms for an indexed field the merge
    wrote none for makes the whole index unopenable. Reachable today:
    `IndexWriter::execute_merge` passes `norms: &[]` unconditionally, so
    `set_norms_field("body")` + commit + merge produced exactly that. Ditto
    `store_term_vectors` when the merge wrote no `.tvd`.
    **Fixed** in `describe_written_files` — `omit_norms` is forced on any
    indexed field the merge wrote no norms for (the same rule, and the same
    justification, the flush path already applies), and `store_term_vectors` is
    cleared when no `.tvd` was written. Tests
    `an_indexed_field_the_merge_wrote_no_norms_for_must_omit_them`,
    `a_field_claiming_term_vectors_loses_the_claim_when_none_were_written`.
    The doc-values **type** is deliberately left alone: a merged field
    declaring `DocValuesType::Numeric` whose values were never supplied through
    `MergeSource` is a caller wiring bug or the sparse-doc-values gap (b10
    finding 21), not something to paper over by rewriting the schema — and
    `IndexWriter::segment_stats` already excludes `.dvd`-bearing segments from
    merging for exactly that reason.

12. **[PERF, fixed] The postings merge materialised every term and re-seeked
    per (term, source), twice with positions.** b10's finding 27.
    `merge_postings` built a `BTreeSet<Vec<u8>>` holding every distinct term of
    the merged field (one heap allocation and one tree insert per (term,
    source)) by traversing every source's dictionary in full, then, per term,
    called `field_terms.postings(&term, ..)` **for every source** — a fresh
    trie-root seek, including for the sources that did not have the term at all
    — and, for a positional field, `field_terms.positions(&term, ..)` on top,
    which re-seeks *and* re-decodes that term's docs and freqs a second time.
    **Fixed** — `merge_postings` is now the k-way streaming merge
    `FieldsConsumer.merge` performs via `MultiTerms`/`MappedMultiFields`: one
    forward `TermCursor` per contributing source, advanced together; at each
    step the smallest current term is the next merged term, and only the
    sources actually standing on it contribute — decoding their postings
    straight off their own cursor position through finding 9's new methods, at
    **zero seeks**. Sources are visited in source order, so the merged doc ids
    stay ascending without a sort (`build_doc_id_maps` gives each source a
    disjoint increasing range). `TermCursor` holds one reusable `Vec<u8>` for
    the current term, because `TermsEnum::try_next`'s borrow cannot outlive the
    next call.
    Java's `docsSeen` bitset replaces the `HashSet<i32>` the old code used to
    count `docCount` — the merged doc id space is dense and its size is known
    up front, so a `FixedBitSet` is both smaller and O(1)-per-posting without
    hashing.
    **9.8x**, and that figure is conservative: the "after" number is derived by
    subtracting a stored-fields-only merge of the same sources from a full
    merge, so it still includes writing `.doc`/`.tim`/`.tip`/`.tmd`, which the
    "before" number does not measure at all.
    Covered by the existing postings-merge tests (positions, offsets, payloads,
    deletions, a source that never saw the field, cross-source schema
    disagreement), all of which pass unchanged, plus the `VerifyMergedSegment`
    end-to-end check that real Lucene finds every term.

13. **[PERF] `Lucene90CompressingTermVectorsWriter.merge`'s BULK path is not
    ported, and cannot be until the term-vectors writer chunks at all.**
    Java's term-vectors merge has the same three-way shape as the stored-fields
    one (`canPerformBulkMerge` + `copyChunks`, same `MatchingReaders`, same
    `tooDirty`, same chunk-header rebase — the only format difference is that
    the token is `(numDocs << 1) | dirtyBit` rather than `<< 2`).
    It is **not tractable in this batch**, and the reason is upstream of the
    merge: `term_vectors::write_best_speed` writes the **entire segment as one
    chunk** (`chunk_docs = docs.len()`, `docBase = 0`, a single non-dirty
    chunk), where Java's writer flushes at 4 096 bytes or 128 documents. There
    is no chunk-appending writer for `copy_chunks` to append to, and no
    multi-chunk source for it to copy from. Porting the fast path therefore
    means first porting real chunking to the term-vectors writer — a
    format-shape change to `lucene-codecs/src/term_vectors.rs` (b7's file)
    large enough to want its own batch and its own fixture verification, and
    one that would change the bytes every existing term-vectors test asserts
    on.
    **What it costs, measured indirectly**: term vectors are merged by
    `merge_term_vectors`, which materialises a `TermVectorsDocument` per merged
    document (`Vec<TermVectorField>`, each with a `Vec<TermVectorTerm>` owning
    its term bytes, positions, offsets and payloads) and then re-encodes all of
    them. That is the same materialise-and-re-encode shape the stored-fields
    VISITOR path had, over a strictly larger per-document payload — and the
    stored-fields measurement above puts that shape at **26x** slower than the
    DOC path and **643x** slower than BULK on the same documents. The
    single-chunk writer also costs read-side random access: fetching one
    document's vectors inflates a compression unit sized for the whole segment.
    **Recorded as a carry-over**, with the chunking prerequisite named.

14. **[PERF, fixed] The merged point list was deep-cloned on the way to the
    writer.** `merge_stored_only_segments` built `WritePointsField { points:
    f.points.clone() }` from a `merged_points_fields` it owned and then dropped
    — a full copy of every point's `Vec<u8>`. Now moved
    (`merged_points_fields.into_iter()`). `write_field` cloned it *again*; that
    clone is gone too on the presorted path (finding 7) and remains only where
    `compute_leaf_plan` genuinely needs ownership.

15. **[INTENTIONAL] A malformed source whose `.fdt` disagrees with its own
    `.fnm` is no longer rejected on the fast paths.**
    `stored_field_number_absent_from_its_own_source_field_infos_is_an_error`
    used to assert that a `MergeSource` whose stored fields reference an
    undeclared field number comes back as `Err`. That check only ever existed
    inside the field-renumbering loop, which is now the VISITOR path alone —
    BULK and DOC copy bytes without parsing field numbers, exactly as Java's do
    (`MatchingReaders` compares `FieldInfos`, not chunk contents; a segment
    whose `.fdt` disagrees with its `.fnm` is a corruption `CheckIndex`
    reports, not something a merge re-derives). The test now declares its field
    with a number the merge has to renumber, which is what puts it on the
    VISITOR path, and says so; the trust boundary is documented on
    `matching_readers` and `write_merged_stored_fields`. A new test,
    `a_matching_deletion_free_source_is_bulk_copied_verbatim`, pins the other
    side.

16. **[PERF] The stored-fields merge no longer holds the merged segment's
    documents in memory.** Both entry points used to build a
    `Vec<Document>` of every surviving document before writing a single byte.
    `write_merged_stored_fields` streams: it walks the `doc_order` both entry
    points already compute, groups consecutive documents from a BULK source
    into one `copy_chunks` call (Java's `while ((sub = docIDMerger.next()) ==
    current)` run detection), and hands DOC/VISITOR documents to the writer one
    at a time. The sorted merge's `merged_docs` is gone the same way; its
    ragged run boundaries fall out of `copy_chunks`' partial-chunk handling.

---

## New verification: real Lucene reads a merged segment

`scripts/verify-write-path.sh` gained a fifteenth case.
`write_merged_segment_fixture` builds three 2 400-document segments (each
spanning two full stored-fields chunks plus a trailing dirty one), deletes
`body:doomed` from segment `_1` through that segment's own real postings, and
merges all three — so `_0` and `_2` take BULK and `_1` takes DOC, in one merge.
`VerifyMergedSegment` then opens the result with `DirectoryReader` and:

- asserts it is a single segment with the right live-document count;
- recomputes every document's expected `id`/`body` independently and compares
  **field by field**, checking the pairing (a bulk-copy boundary error shows up
  as a document holding another document's fields) and the multiset (a document
  copied twice while another is dropped) rather than an order the merge policy
  is free to choose;
- checks the merged postings (`body:shared` in every document, `body:doomed`
  surviving in exactly the two segments it was not deleted from);
- runs `CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS`.

This is what caught findings 10 and 11 — both of which every Rust-side test and
`CheckIndex` itself were blind to.

---

## Tier 2 review findings (`quality-reviewer`, run on this batch's diff)

The reviewer independently verified, line by line against the Java, that
`copy_chunks` is faithful (including the `.fdx` bookkeeping against
`FieldsIndexWriter.finish`), that `presorted_leaf_plan` is equivalent to
`compute_leaf_plan` **in every tree shape** and not just the sampled ones, and
that `try_current_postings`/`try_current_postings_and_positions` are safe
between `try_next` calls (`decode_meta_data` is a `metaDataUpto`-bounded
catch-up loop, so the second call is a no-op). Six further items came out of it;
all are resolved.

17. **[CORRECTNESS] No `checkIntegrity` on a source before its bytes were
    copied — the bulk path could launder corruption into a permanently valid
    segment.** Java runs `reader.checkIntegrity(mergeState.oneMerge)` —
    `CodecUtil.checksumEntireFile` — on every source *before* `getMergeStrategy`
    picks a path. This port did not, and `stored_fields::open` deliberately
    only calls `retrieve_checksum`, which validates the footer's *shape* and
    not the CRC (there is a test asserting exactly that:
    `retrieve_checksum_does_not_detect_payload_corruption`).
    That was tolerable while every merge decompressed and re-encoded every
    document — a bit flip surfaced as an LZ4/DEFLATE decode error. It is not
    tolerable now: BULK copies a source's compressed bytes verbatim and then
    writes a **freshly computed, valid footer** over them, so a corrupt source
    would become a merged segment that passes every checksum from then on.
    This is the hazard behind Java's "bulk merge is scary: its caused
    corruption bugs in the past" comment.
    **Fixed** — new `StoredFieldsReader::check_integrity()`
    (`codec_util::check_whole_file_footer` over `fdt[..max_pointer]`), called
    on every source in `write_merged_stored_fields` before the strategy is
    chosen, exactly where Java calls it. Test
    `a_source_whose_fdt_fails_its_own_checksum_is_never_bulk_copied` flips one
    byte of a source's compressed payload — leaving every length, pointer and
    footer field intact, i.e. precisely the corruption `retrieve_checksum`
    cannot see — and asserts the merge refuses it. The cost is a full CRC of
    each source `.fdt` per merge, which is why the BULK figure above is 520x
    rather than the 643x the same benchmark read before this check existed.

18. **[MISSING] The two `.fdx`/`.fdt` disagreement branches in `copy_chunks`
    had no test.** The guards existed and their comment correctly called them
    "how a bad bulk copy announces itself", but nothing exercised them — an
    untested guard against the batch's own headline failure mode.
    **Fixed** — three corruption tests, each mutating a valid two-chunk
    segment's `.fdt` with a same-length vint substitution so `open`'s
    `maxPointer`-vs-file-length cross-check still passes and the corruption
    really has to be caught by `copy_chunks`:
    `a_chunk_header_whose_doc_base_disagrees_with_the_index_is_rejected`
    (Java's `base != docID`),
    `a_chunk_header_claiming_no_documents_is_rejected` (a zero-document chunk
    would make the copy loop spin without advancing), and
    `a_chunk_claiming_more_documents_than_the_requested_range_is_rejected`
    (Java's `docID > toDocID`). All three passed on the first run.

19. **[CORRECTNESS] `copy_chunks` enforced its safety precondition only in
    debug builds.** It is a `pub` method on a `pub` type in a library crate,
    guarded by `debug_assert!(self.can_bulk_copy(reader))`. Java can afford an
    `assert` there because `copyChunks` is private with one caller; here a
    release-mode caller copying from a reader with a different `chunkSize`
    would get a segment whose `sliced` chunks are re-split at the wrong
    boundary — silently wrong documents.
    **Fixed** — a real `Error::BulkCopyNotPermitted` carrying which of the
    three conditions failed, plus `Error::InvertedDocRange` so a `from > to`
    range no longer reports itself as "doc 5 is out of range (maxDoc=10)".
    Covered by `copy_chunks_rejects_an_out_of_range_document_range`.

20. **[MISSING] A broken intra-doc link to the function this batch deleted.**
    `write_best_compression`'s doc comment still pointed at
    `[`write_chunked`]`, which no longer exists — and said something now
    factually wrong (it shares `StoredFieldsWriter`, not that function).
    `rustdoc::broken_intra_doc_links` is warn-by-default and is caught by none
    of `cargo fmt`/`clippy`/`test`. **Fixed**; a
    `cargo doc -p lucene-codecs -p lucene-index --no-deps` run now reports no
    *unresolved* link in any file this batch touched (the three that remain in
    `merge.rs`/`points.rs` are pre-existing and predate this batch). Adding a
    rustdoc pass to the gate is recorded as a carry-over rather than done here,
    because the workspace has many pre-existing broken links in files other
    batches own.

21. **[INTENTIONAL, now documented] `describe_written_files` tolerates a real
    loss for norms and term vectors, and the doc comment claimed more principle
    than it had.** The reviewer's point: the function refuses to rewrite the
    doc-values *type* on the grounds that a missing value is "a caller wiring
    bug, not something to paper over", then does exactly that for
    `omit_norms`/`store_term_vectors` — silently dropping norms changes every
    BM25 score in the merged segment with no signal anywhere.
    That is right, and the asymmetry is now stated rather than dressed up. What
    the comment previously missed is that the doc-values case really is
    different for a reason: `IndexWriter::segment_stats` excludes any
    `.dvd`-bearing segment from merging outright, precisely so doc values
    cannot be lost this way. There is no equivalent exclusion for norms.
    **Resolved as documented-and-carried-over, not as an error**, because
    turning it into one would mean this port cannot merge *any* indexed segment
    until `IndexWriter::execute_merge` opens a source's norms — and
    `execute_merge` is `c3-writer-lifecycle`'s file this batch must not touch.
    Note the pre-`c4` behaviour was strictly worse, not better: the merged
    `.fnm` kept `omit_norms = false` with no `.nvm` written, which makes
    `DirectoryReader.open` throw. The carry-over below names the fix.

22. **[MISSING] Two tests could not catch the regression they were named
    after.** `matching_readers_is_the_identity_field_number_test` re-implemented
    the predicate inline instead of calling `matching_readers`, so changing that
    function to `map.get(&f.number).is_some()` — which would make every
    renumbered source eligible for BULK, the exact "values landing under the
    wrong field name" failure it exists to prevent — would have kept it green.
    And `presorted_plan_matches_the_general_plan_byte_for_byte` used only
    strictly increasing values, where the equivalence argument rests on
    `compute_leaf_plan`'s `sort_by` being *stable*; an unstable sort would have
    passed too.
    **Fixed** — the first now builds real `MergeSource`s, calls
    `matching_readers` directly, and additionally asserts the strategy each
    source ends up with. The second gained
    `presorted_plan_matches_the_general_plan_with_duplicate_values`, which puts
    leaf boundaries *inside* runs of equal values (run 3 against leaf size 8,
    plus an all-equal field) and compares the plans and the emitted bytes.

23. **[PERF] Recorded, not fixed: `flush` omits the empty compression unit Java
    always writes.** For a chunk whose documents are all empty
    (`total_length == 0`) Java still emits `vint(0) vint(0) doCompress(…, 0, …)`
    — three zero bytes — where this port emits nothing. Carried over verbatim
    from the old `write_chunked`, so not introduced here, and this port's
    reader short-circuits on `total == 0` either way. Flagged because the code
    moved past it: no fixture covers an all-empty-document chunk, so the
    divergence is assumed benign rather than proven so.

---

## Verdict

23 findings: **4 CORRECTNESS (all fixed), 3 MISSING (all fixed), 11 PERF
(8 fixed and measured, 3 recorded), 5 INTENTIONAL.**

**All three of b10's PERF carry-overs are addressed; two closed outright, one
partially, and one new item opened.**

- b10 #24 (stored-fields bulk copy) — **closed.** All three of Java's merge
  strategies ported, verified by real Lucene against a merged segment, 643x /
  23x / 26x.
- b10 #27 (streaming postings merge) — **closed.** Real k-way `TermsEnum`
  merge with zero-seek postings decoding, 9.8x.
- b10 #29 (`BKDWriter.merge`) — **closed for the case Java optimizes** (and one
  case it does not), 2.5x.
- Term vectors' bulk merge — **not tractable in this batch**; blocked on the
  term-vectors writer emitting one chunk per segment. Cost quantified above
  (finding 13).

Four CORRECTNESS defects were fixed on the way: 10 and 11 (a merged segment's
postings invisible to real Lucene, and a merged `.fnm` promising norms that
made the index unopenable — both invisible to every in-port test *and* to
`CheckIndex`), 17 (no source checksum before a byte-copy merge, so the bulk
path could launder corruption into a permanently valid segment) and 19 (the
bulk path's safety precondition enforced only in debug builds).

## Carry-over items raised by this batch

- [ ] **Term-vectors chunking, then its bulk merge.**
      `term_vectors::write_best_speed` writes the whole segment as one chunk;
      Java flushes at 4 096 bytes / 128 documents. Porting real chunking is the
      prerequisite for `Lucene90CompressingTermVectorsWriter.merge`'s BULK
      path (which is otherwise structurally identical to the stored-fields one
      already ported here) and would also fix random-access reads, which
      currently inflate a segment-sized compression unit to fetch one
      document's vectors. Owner: a `lucene-codecs`/term-vectors batch.
      (Finding 13.)
- [ ] **`IndexWriter::execute_merge` supplies no norms**, so an automatic merge
      silently drops them and changes every BM25 score in the merged segment.
      Findings 11 and 21 make the merged `.fnm` honest about that (which is
      what keeps the index openable at all), but the loss itself is real and
      the fix belongs in `index_writer.rs` — open each source's `.nvm`/`.nvd`
      and populate `MergeSource::norms`, the way `execute_merge` already does
      for postings and term vectors. Owner: `c3-writer-lifecycle` or a
      successor. Doc values are a different case and are already safe:
      `segment_stats` excludes `.dvd`-bearing segments from merging entirely.
- [ ] **`merge_sorted_stored_only_segments` writes no postings**, so
      `describe_written_files` is called there with an empty postings list. Not
      newly introduced; recorded because the `.fnm` now states it explicitly.
- [ ] **A rustdoc pass belongs in the gate.**
      `rustdoc::broken_intra_doc_links` is warn-by-default and is caught by
      none of `cargo fmt`/`clippy`/`test`; this batch shipped one broken link
      (finding 20) that survived a green Tier 1 gate. Adding
      `RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --workspace
      --no-deps` to `AGENTS.md`'s command table needs the pre-existing broken
      links in `doc_values.rs`, `for_util.rs` and others cleaned up first, so
      it is not this batch's to turn on.
- [ ] **An all-empty-document stored-fields chunk has no fixture**, so this
      port's omission of Java's three-zero-byte empty compression unit is
      assumed benign rather than proven. (Finding 23.)
- b10's still-open items are unchanged: `MergeSource` cannot carry per-source
  `min_version`/`has_blocks`; zero-doc merges should be dropped by
  `IndexWriter::apply_merge`; sparse doc-values/norms merges stay a hard error
  until `lucene-codecs` can write `IndexedDISI`-backed sparse fields.

## Concurrency

`merge.rs` and `merge_policy.rs` were this batch's; `index_writer.rs`,
`segment_writer.rs`, `indexing_chain.rs` and `segment_infos.rs` were left alone
(`c3-writer-lifecycle`'s). `blocktree.rs` is `c1-lazy-blocktree`'s and received
only the two additive `TermsEnum` methods in finding 9, at the end of an
existing `impl` block.

The workspace build broke twice mid-batch inside
`crates/lucene-codecs/src/vectors.rs`/`hnsw.rs`, a file another in-flight batch
was creating, and `cargo clippy` failed on three lints in `hnsw.rs` for a
while after that. Per protocol this batch waited and retried rather than
editing it; both cleared, and the gate is green as recorded above.
