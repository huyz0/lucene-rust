# c22-sorted-merge

Follow-up batch closing the half of index sorting `c17-index-sort` could not
wire, plus `c10-vectors-wiring`'s handoff.

c17 made `IndexWriter::flush` produce sorted segments and proved it against
real Lucene, but stopped at the merge, in its own words: *"`merge_sorted_
stored_only_segments` writes no postings and no vectors, so routing
`execute_merge` through it would trade a lost sort for lost data."* Its
containment was to make `segment_stats` refuse to offer a sorted segment to
the merge policy at all, so a sorted index never auto-merged.

That containment is now unnecessary. **There is one merge**, `merge_segments`,
and an index sort changes nothing about it except the order documents come out
in.

Java counterparts (Lucene 10.5.0 at `/home/tuong/work/lucene-10.5.0`; every
citation re-checked against the pinned tree, per `c18-version-audit`):

- `index/SegmentMerger.java` (`merge()`'s per-format sequence)
- `index/MergeState.java` (`buildDocMaps`, `buildDeletionDocMaps`, `DocMap`,
  `needsIndexSort`)
- `index/MultiSorter.java` (`sort`, the `LeafAndDocID` priority queue and its
  tie-breaks)
- `index/DocIDMerger.java` (`SequentialDocIDMerger` vs `SortedDocIDMerger`,
  and `of(subs, indexIsSorted)`)
- `index/Sorter.java`, `index/IndexSorter.java` (the comparator)
- `index/IndexWriter.java` (`validateIndexSort`, `mergeMiddle`)
- `codecs/lucene90/compressing/Lucene90CompressingStoredFieldsWriter.java`
  (`merge`, `getMergeStrategy`, `copyChunks`, `copyOneDoc`) and its term-vectors
  twin
- `codecs/lucene99/Lucene99FlatVectorsWriter.java`
  (`mergeOneFlatVectorField`), `codecs/lucene99/Lucene99HnswVectorsWriter.java`
  (`mergeOneField`, `buildAndWriteGraph`), `util/hnsw/IncrementalHnswGraphMerger.java`
- `codecs/lucene99/Lucene99SegmentInfoFormat.java` (`write`'s `si.addFile`)
- `codecs/lucene90/Lucene90DocValuesConsumer.java` (`addNumericField`,
  `writeValues`)

Totals: **26 findings** -- 11 CORRECTNESS (all fixed), 8 MISSING (7 fixed, 1
recorded), 3 PERF (2 fixed, 1 measured), 4 INTENTIONAL. Six of them (21-26)
came out of the Tier-2 review; three were gating, and two of those were
defects this batch itself introduced.
(Numbered in file order, so the last file's finding comes before the
merge-policy one.)

`scripts/verify-write-path.sh`: this batch adds exactly one case (**20 -> 21**;
c17 recorded 20, and every pre-existing case is green alongside the new one).
The final run of the session reports **22/22**, because a concurrent batch
added a `VerifyPositionsSegment` case while this one was in flight. Run, not
assumed, four times across the batch.

---

## crates/lucene-index/src/merge.rs

Java counterparts: `index/SegmentMerger.java`, `index/MergeState.java`,
`index/MultiSorter.java`, `index/DocIDMerger.java`, and the per-format
`merge`/`mergeOneField` entry points listed above.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `merge_segments` (new) | `SegmentMerger.merge()` | the one merge; `sort_fields: Option<..>` is `MergeState.buildDocMaps`'s two branches |
| `merge_stored_only_segments` | `SegmentMerger.merge()` with `segmentInfo.getIndexSort() == null` | now a wrapper (finding 1) |
| `merge_sorted_stored_only_segments` | ...with a sort | now a wrapper (finding 1) |
| `sorted_doc_order` (extracted) | `MultiSorter.sort` | equivalent; linear head scan instead of a `PriorityQueue`, same tie-breaks (finding 11) |
| `concat_doc_order` | `MergeState.buildDeletionDocMaps` | identical |
| `build_doc_id_maps` | `MergeState.DocMap[]` | **was divergent (finding 3)**; now derived from `doc_order` |
| `compare_heads` | `MultiSorter`'s `lessThan` | identical, including `readerIndex` then `docID` tie-breaks |
| `write_merged_stored_fields` | `Lucene90CompressingStoredFieldsWriter.merge` | identical shape; **PERF defect fixed (finding 9)** |
| `write_merged_term_vectors` | `Lucene90CompressingTermVectorsWriter.merge` | same |
| `merge_{numeric,binary,sorted,sorted_numeric,sorted_set}_doc_values` | `DocValuesConsumer.merge*` | **widened to field lists (finding 5)**; NUMERIC may now be sparse (finding 6), all-missing included (finding 21) |
| `merge_norms` | `NormsConsumer.mergeNormsField` | unchanged, still one field per merge (finding 13) |
| `merge_postings` | `FieldsConsumer.merge` + `MappingMultiPostingsEnum` | **was divergent (finding 4)**; now re-orders a term's docs when the sources interleave, which is `DocIDMerger`'s sorted mode |
| `merge_points` | `Lucene90PointsWriter.merge` / `BKDWriter.merge` | doc map fixed (finding 3); the value-ordered stream is unaffected by the document order |
| `merge_vectors` (new) | `SegmentMerger.mergeVectorValues` -> `Lucene99HnswVectorsWriter.mergeOneField` | **added (finding 7)**, c10's five-step handoff |
| `describe_written_files` | `IndexWriter`'s `fields_with_per_field_attributes` equivalent | **widened (finding 8)** |
| `take_permuted` (new) | -- | not-in-Java; the helper finding 4's re-ordering needs |
| `MergeOptions` (new) | the codec's configured `M`/`beamWidth` | not-in-Java as a type; Java reads them off the `KnnVectorsFormat` |

Java with **no** Rust counterpart: `MergeState`'s `SegmentReader`-driven
construction (this port's `MergeSource` is caller-supplied by design, b10);
`MergeState.needsIndexSort`'s "the sources happen to already be in order"
short-circuit, where `MultiSorter.sort` returns `null` and Java falls back to
the deletion doc maps -- here the k-way merge *is* the deletion doc map when
the sources do not interleave, so there is nothing to skip; `CachingMergeContext`.

### Findings

1. **[CORRECTNESS -> fixed]** *The sort-preserving merge wrote no postings, no
   points and no vectors.* c17's blocker. Two entry points had grown apart:
   `merge_stored_only_segments` wrote every format and destroyed the sort;
   `merge_sorted_stored_only_segments` preserved the sort and dropped three
   formats. Neither is usable for a sorted index, and *which* one you call
   decides which kind of silent loss you get.
   Fixed by collapsing them into one `merge_segments` whose only branch is
   `doc_order`, with the two old names kept as wrappers so no caller changed.
   That is Java's shape: `SegmentMerger.merge()` runs the same per-format
   sequence either way, and `MergeState.buildDocMaps` is where the sort enters.
   The property it buys is structural: a format cannot be written by one entry
   point and forgotten by the other, because there is only one.
   *Tests*: `an_automatic_merge_preserves_the_index_sort_across_every_format`,
   `a_merge_honours_every_tier_of_a_multi_field_reverse_sort`,
   `a_sorted_merge_drops_deleted_documents_from_every_format`, and the new
   `VerifySortedSegment` case below.

2. **[CORRECTNESS -> fixed]** *A merged `.si` never listed itself.*
   `Lucene99SegmentInfoFormat.write` does `si.addFile(fileName)` *before*
   writing, so a real Lucene `.si` is in its own file list. The flush path
   (`segment_writer`) did this; the merge did not, for every merged segment
   this port has ever written -- `files: files.clone()` ran *before*
   `files.push(si_name)`.
   The consequence that matters is `IndexFileDeleter`, which reference-counts
   from exactly that set: a merged segment's own `.si` was a file nothing held
   a reference to.
   It survived five batches of merge work because **nothing looks**. Real
   `CheckIndex` does not verify the self-reference (it only counts
   `info.files()`), so `VerifyMergedSegment` stayed green; this port's own
   `check_index` *does* (`si.files_lists_itself`, added by c9 from
   `Lucene99SegmentInfoFormat.write` rather than from `CheckIndex`), and no
   test had ever run it over a merged segment. The four new end-to-end merge
   tests do, which is how it surfaced.
   *Test*: every one of the four new end-to-end merge tests runs
   `check_index::check_directory` over the merged segment.

3. **[CORRECTNESS -> fixed]** *`MergeState.DocMap` was derived from the
   concatenation rule, not from the merge.* `build_doc_id_maps` computed
   "source `i`'s live docs land immediately after source `i-1`'s" from
   `per_source_live_ids` alone. `merge_postings`, `merge_points` and (now)
   `merge_vectors` look documents up through it, while stored fields, doc
   values, norms and term vectors *walk* `doc_order`. Under a sorted merge the
   two are different mappings, so postings and points would have addressed the
   concatenated space while every other format addressed the sorted one --
   every file valid, every doc id in range, and terms attached to the wrong
   documents. Now inverted from `doc_order` itself, and sized by each source's
   `max_doc` rather than by its last live doc (a vector merge looks up
   documents past that point).
   *Test*: `the_doc_id_maps_invert_the_doc_order_for_both_merge_orders`, which
   asserts the inverse property in both directions for both orders, that
   unnamed documents map to nothing, and that the map is increasing *within* a
   source -- the property findings 4 and 19 both rest on.

4. **[CORRECTNESS -> fixed]** *`merge_postings` concatenated a term's postings
   in source order.* Correct only because merged doc ids occupied disjoint,
   increasing per-source ranges -- which is exactly what a sorted merge stops
   being true. `postings_writer` delta-encodes doc ids, so an interleaved term
   would write a negative delta; and a pair of sources whose deltas happened to
   stay positive would encode postings against the wrong documents. Java uses
   `DocIDMerger`'s `SortedDocIDMerger` (a heap on the mapped doc id) for
   exactly this. Fixed by re-ordering the four parallel per-document lists
   (docs, positions, offsets, payloads) together when the concatenation is not
   already ascending -- one linear scan, which moves nothing on the unsorted
   path.
   *Tests*: `take_permuted_moves_each_element_to_its_named_position` (pinned
   with a 3-cycle, since an involution cannot tell a permutation from its
   inverse -- the failure mode c17 recorded for `permute_in_place`), and the
   end-to-end tests, each of which asserts that every document's *unique* term
   resolves to exactly that document's merged doc id.

5. **[MISSING -> fixed]** *One doc-values field per type, one type per merge.*
   Six error variants (`TooMany{Numeric,Binary,Sorted,SortedNumeric,SortedSet}
   DocValuesFields`, `MultipleDocValuesTypesInOneMerge`) enforced a limit that
   came from the *writers* -- `write_single_dense_*` each produce a whole
   `.dvm`/`.dvd`/`.dvs`, so two of them in one merge overwrote each other.
   `doc_values::write_dense_fields` had removed that limitation and the merge
   had never taken it up. It matters here because **a multi-tier index sort is
   inexpressible without it**: a second sort tier is a second NUMERIC column.
   Every merged doc-values field of every type now goes into one triple, which
   is what a real `Lucene90DocValuesFormat` segment is. The six variants are
   gone.
   *Tests*: the five `two_*_doc_values_fields_land_in_one_merged_dvm` cases and
   the two `*_share_one_merged_dvm` ones -- the former rejection tests,
   rewritten to read both columns back through the unmodified reader stack
   rather than to assert an error.

6. **[MISSING -> fixed]** *A sparse NUMERIC column could not be merged at
   all.* A live document with no value for the field returned
   `DocValuesFieldMissingInSource`. That is `SortField.setMissingValue`'s
   normal case, and c17's own fixture has 54 documents with no `rank`, so
   merging the very index c17 verified was impossible. Now the merged column
   is written sparsely through the same `IndexedDISI` + values body
   `Lucene90DocValuesConsumer.writeValues` uses; a column no merged document
   has a value for is dropped entirely and the merged `.fnm` does not claim
   it. A source that never *declared* the field is still a hard error --
   schema mismatch is not sparsity.
   *Test*: the `VerifySortedSegment` merged case, whose `rank` column is
   sparse and whose absences are checked per doc id by real Lucene's
   `NumericDocValues.advanceExact`.

7. **[MISSING -> fixed]** *Vectors were not merged.* c10's handoff, verbatim
   in structure: open each source's `.vec`/`.vemf` (and `.vem`/`.vex` when it
   has a graph), skip any source whose own `FieldInfo` gives the field
   `vector_dimension == 0` (`hasVectorValues` in `buildAndWriteGraph`), merge
   the flat store *first* because that is what defines the merged ordinal
   space, reopen exactly those bytes, and merge the graphs against them
   through `hnsw_vectors::merge_one_field`. One deviation from the recipe: a
   field every one of whose vectors was deleted is omitted rather than written
   empty, and its `vector_dimension` is zeroed -- the flush path's rule.
   *Tests*: the four end-to-end merge tests assert every vector component per
   doc id and that the merged graph's node count is the merged one; the
   `VerifySortedSegment` merged case runs a real `KnnFloatVectorQuery` over
   the merged graph.

8. **[MISSING -> fixed]** *`describe_written_files` did not describe two of
   the formats, and justified the gap with a guard that no longer exists.* It cleared `omit_norms` and `store_term_vectors` for data the
   merge did not write, but left `doc_values_type` and `vector_dimension`
   claiming data that was not there, and never stamped
   `PerFieldKnnVectorsFormat`. Both are the silent shape:
   `PerFieldDocValuesFormat`/`PerFieldKnnVectorsFormat` register no reader for
   a field with no format attribute, so the field reads back as
   doc-values-capable or vector-capable and yields nothing, and real
   `CheckIndex.testDocValues` dereferences the producer that was never
   registered. `IndexWriter::fields_with_per_field_attributes` already had
   both rules at flush time; this is them at merge time. (This is also c17's
   finding 14 for the merge path.)
   The doc comment also said the doc-values type was "left alone, and that
   asymmetry **is** principled: `segment_stats` excludes any `.dvd`-bearing
   segment from merging outright" -- both halves false as of findings 15 and
   18, and it was the written justification for the caller-side guard findings
   22 and 23 turned out to be missing. Rewritten to say what actually holds:
   **every** rewrite here tolerates a loss, and the guard lives on the caller
   side -- `execute_merge` opens every format its own flush can write, and
   `segment_stats` withholds the one case it cannot round-trip.

9. **[PERF -> fixed, 152x]** *A sorted merge decompressed a whole stored-fields
   chunk per document.* `StoredFieldsWriter::copy_chunks` allocated a **fresh**
   `ChunkCursor` on every call for the partial chunks at each end of the run.
   With no sort that is invisible: one call per source, one run of `0..maxDoc`.
   With a sort the sources interleave, every run is a document or two, and
   every one of them re-decompressed the chunk the previous run had just
   decompressed. Java does not have the problem because `copyOneDoc` reads
   through the *reader's* own cached `BlockState`, which persists across
   `copyChunks` calls.
   Fixed by adding `copy_chunks_with_cursor` to
   `stored_fields::StoredFieldsWriter` and `term_vectors::TermVectorsWriter`
   (additive; `copy_chunks` is now a one-line wrapper) and passing the
   per-source cursor `write_merged_stored_fields`/`write_merged_term_vectors`
   already keep for their DOC path. **4 segments x 20 000 documents: 2 004 ms
   -> 13.2 ms.**

10. **[PERF, measured]** *What the sort costs a merge.* See Measurements: after
    finding 9, an index-sorted merge of the same sources costs **13.2 ms
    against the unsorted merge's 0.9 ms (14.7x)**. That is the byte-copy fast
    path becoming illegal, not a regression: `copy_chunks` copies *compressed*
    chunk bytes, which encode a contiguous run of documents, and a sort
    interleaves the sources so no chunk lies entirely inside one run. The
    resulting cost is c4's DOC path (which c4 measured at 26x faster than the
    pre-c4 merge), and the strategy selection is unchanged from Java's, which
    also picks BULK and lets the runs collapse.

11. **[INTENTIONAL]** *A linear head scan, not a `PriorityQueue`.*
    `MultiSorter` uses a heap over `leafCount` entries; `sorted_doc_order`
    scans the (few) source heads per step. `max_merge_at_once` defaults to 10,
    so the log factor buys nothing against the constant -- the same reasoning
    `merge_point_streams` already records for `BKDWriter.merge`'s own queue.

12. **[INTENTIONAL]** *No bulk-copy path and no `MatchingReaders` for vectors.*
    Lucene has neither: `mergeOneFlatVectorField` always re-writes the data
    file, because the merged ordinal space is new. What this port adds instead
    is a `memcpy` per unbroken run of surviving ordinals, which Java does not
    have (it re-encodes each vector through a `ByteBuffer`).

13. **[MISSING, recorded]** *Norms still take one field per merge*
    (`Error::TooManyNormsFields`). `norms::write_single_dense_field` writes one
    field's whole `.nvm`/`.nvd` pair and there is no multi-field norms writer
    to widen onto, the way `write_dense_fields` existed for doc values.
    `IndexWriter::set_norms_field` is itself single-field, so nothing this port
    can flush reaches the limit. Closing it is a `lucene-codecs` change
    (`write_dense_fields`' shape, for `.nvm`/`.nvd`), not a merge change.

21. **[CORRECTNESS -> fixed, from the Tier-2 review]** *An all-missing merged
    NUMERIC column was dropped instead of written -- and for a sort tier that
    produces a segment whose `.si` lies.* A defect this batch introduced with
    finding 6: `merge_numeric_doc_values` pushed nothing when no merged
    document had a value, `describe_written_files` (finding 8) then zeroed the
    field's `DocValuesType`, and the merged `.si` went on declaring the sort
    over it. Real Lucene's `DocValues.getNumeric` **throws** for a field whose
    `FieldInfo` exists but declares no doc values (`DocValues.checkField`), so
    `CheckIndex.testSort` fails rather than degrading; and this port could
    never merge that segment again, because `read_sort_keys` returns
    `MergeSortColumnMissing` and `auto_merge` propagates it straight out of
    `commit`, leaving the index permanently un-committable.
    Java has no such branch: `SegmentMerger.mergeDocValues` calls the consumer
    for every field the merged `FieldInfos` gives a type, and
    `Lucene90DocValuesConsumer.writeValues` records an all-missing column as
    `docsWithFieldOffset = -2`. So does this now -- the special case is gone,
    not special-cased for sort fields.
    *Test*: `a_sort_tier_no_merged_document_has_a_value_for_is_still_written`,
    which merges three segments none of which has a value for the second tier
    and asserts the merged `.fnm` still declares it, the column reads back
    all-missing, `check_index` is clean, **and the segment can be merged
    again**. Verified to fail without the fix (`doc_values_type` comes back
    `None`).

22. **[CORRECTNESS -> fixed, from the Tier-2 review]** *An automatic merge
    dropped every doc-values type except NUMERIC.* Finding 15 opened each
    source's `.dvm` and finding 18 dropped `segment_stats`' `.dvd` exclusion,
    but `execute_merge` built merge sources only from `meta.numeric` -- so a
    BINARY / SORTED / SORTED_NUMERIC / SORTED_SET column (all four reachable
    from `set_doc_values_field`/`add_doc_values_field` through
    `collect_dense_column`) was merged away, with `describe_written_files`
    zeroing its type to keep the `.fnm` honest. A valid, `CheckIndex`-clean
    segment with the data gone: exactly the loss the exclusion existed to
    prevent, reintroduced by removing it. All five lists are built now, and a
    `debug_assert!` pins "every entry a source's `.dvm` declares reaches the
    merge" so a sixth type cannot be forgotten quietly.
    *Test*: `an_automatic_merge_carries_every_doc_values_type_through`, which
    checks both columns per document against the stored `id` that identifies
    it. Verified to fail with the BINARY list stubbed back out.

23. **[CORRECTNESS -> fixed, from the Tier-2 review]** *An automatic merge
    dropped positional postings.* `execute_merge` filtered its merge-time
    postings fields to `Docs`/`DocsAndFreqs`, which was accurate when this
    writer could flush nothing else. It can now
    (`DocsAndFreqsAndPositions(AndOffsets)`), and `segment_stats` offers those
    segments -- so a merged segment carried no postings for the field while
    `describe_written_files` left its `index_options` declaring them: an
    indexed field with no registered postings producer, which reads back as
    having no terms and raises nothing. The filter now matches exactly what
    `set_postings_field` accepts, and each source's `.pos`/`.pay` are opened
    alongside its `.doc` (`merge_postings` already handled both).
    *Test*: `an_automatic_merge_carries_positional_postings_through`, which
    reads every merged document's positions back through
    `blocktree::FieldTerms::positions`.

24. **[CORRECTNESS -> fixed, from the Tier-2 review]** *Every merge dropped
    `has_blocks`.* `merge_segments` wrote `has_blocks: false` unconditionally.
    `IndexWriter.mergeMiddle` ORs it across `merge.segments`, and this port's
    `add_documents` sets it, so a merge of block-bearing segments produced a
    segment that reads back perfectly and silently invalidates every
    parent/child join query against it. Pre-existing (b10 recorded that
    `MergeSource` cannot carry it), but this batch's own carry-over note
    claimed the situation was unreachable, which is true only for a *sorted*
    merge. `MergeOptions::has_blocks` carries it, set by `execute_merge` from
    each source's `.si` -- an option rather than a `MergeSource` field because
    Java reads it from `SegmentCommitInfo` in the writer too, not in
    `SegmentMerger`.
    *Test*: `an_automatic_merge_keeps_has_blocks`, which merges two
    block-bearing commits with a plain one (so the flag must be ORed, not
    copied from source 0) and asserts both the flag and each block's
    contiguity in the merged segment.

25. **[PERF -> fixed, from the Tier-2 review]** *`numeric_value` called once
    per document in two whole-column walks.* `doc_values::numeric_value` is a
    free function that allocates a fresh `DisiCursor` and re-walks the sparse
    docs-with-field region from its start on every call -- `NumericReader`'s
    own doc comment names *a sort* as the caller that must not do that.
    `read_sort_keys` walks `0..max_doc` once per tier per source and
    `merge_numeric_doc_values` walks the whole `doc_order`; both now hold one
    `NumericReader` per source. It is finding 9's defect shape one module
    over, and both call sites are forward-only in the source's own doc ids
    (`build_doc_id_maps` is monotone within a source), which is exactly the
    access pattern the cursor is built for.

26. **[INTENTIONAL, from the Tier-2 review]** *`merge_segments` reports a
    mis-shaped sort-key table instead of panicking.* The sorted entry point
    had `assert!`/`assert_eq!` on `sort_fields`/`per_source_keys` since b10,
    in a `pub fn`. `Error::EmptySortFields` and `Error::SortKeysWrongLength`
    replace them, which is the call the rest of this module already makes --
    and the one `read_sort_keys`' own doc comment, written a few hours earlier
    in this same batch, argues for.
    *Test*: `a_mis_shaped_sort_key_table_is_reported_rather_than_panicking`,
    including the shape a caller who filtered deletions first would naturally
    build (one entry per *live* document rather than per document).

### Verdict

Swept clean. One merge, one document order, every format through it. Open:
finding 13.

---

## crates/lucene-index/src/index_writer.rs (`execute_merge` / `segment_stats` only)

Owned by c7/c17 across the sweep; this batch's edits are scoped to the two
methods that decide *what* gets merged and *how*, plus the error variants they
need. Java counterparts: `IndexWriter.{mergeMiddle,validateIndexSort}`,
`MergeState`'s reader assembly, `MultiSorter`'s `getComparableValues`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `execute_merge` | `IndexWriter.mergeMiddle` + `MergeState`'s constructor | **widened (findings 14-17)**: doc values, norms and vectors are opened; the sort is validated and its keys read |
| `read_sort_keys` (new) | `IndexSorter.LongSorter.getComparableValues` | added (finding 17) |
| `segment_stats` | `updatePendingMerges` -> `MergePolicy.findMerges` | **two exclusions dropped (finding 18)**, one kept |

### Findings

14. **[CORRECTNESS -> fixed]** *An automatic merge silently dropped norms.*
    `c4-merge-fastpath`'s open carry-over, unchanged since: `execute_merge`
    passed `norms: &[]`, so a merged segment had none, and
    `describe_written_files` then cleared `omit_norms` to keep the index
    openable -- which is what made the loss survivable and therefore silent.
    Every BM25 score in a merged segment was wrong. `execute_merge` now opens
    each source's `.nvm`/`.nvd` and merges them.
    *Test*: `assert_every_format_agrees` reads every merged document's norm
    back and checks it against the field length that document was written
    with; it runs in all four end-to-end merge tests.

15. **[MISSING -> fixed]** *An automatic merge could not carry doc values, so
    doc-values-bearing segments were never merged.* `segment_stats` excluded
    every `.dvd`-bearing segment from the candidate pool, which was the right
    call while `execute_merge` opened no doc values, and which (because
    `set_index_sort` makes a sort field's column mandatory) is also what kept
    sorted segments out. All five types are now opened and merged -- the first
    cut of this fix opened all five but wired only NUMERIC, which the Tier-2
    review caught as finding 22.
    *Test*: `an_automatic_merge_carries_doc_values_through_instead_of_dropping_them`,
    which asserts not that the merge happened but that each merged document's
    stored `id` and its doc-values `score` still describe the same document.

16. **[MISSING -> fixed]** *An automatic merge could not carry vectors.* Same
    shape; `execute_merge` opens the flat pair and the graph pair (tolerating a
    segment that has the first and not the second, which is what a
    sub-`HNSW_GRAPH_THRESHOLD` flush writes), and passes its configured
    `hnsw_m`/`hnsw_beam_width` through `merge::MergeOptions`.

17. **[MISSING -> fixed]** *No `validateIndexSort` at merge time, and no sort
    keys.* `execute_merge` reads each source's `.si`, requires every one to
    declare the same sort (or none), and refuses otherwise --
    `Error::MergeSortDisagreement`, Java's
    `IllegalArgumentException("cannot change index sort ...")`. Each tier's
    per-document key is then read out of **the very NUMERIC column the merged
    segment will carry**, so the order the merge imposes and the column
    `CheckIndex.testSort` re-derives it from are one fact, exactly as at flush
    time. A segment that declares a sort but has no column for a tier is
    `Error::MergeSortColumnMissing` rather than an all-missing read: a merge
    that quietly ordered every one of its documents by a sentinel would produce
    a `CheckIndex`-clean segment in the wrong order.
    *Test*: `a_merge_of_segments_that_disagree_about_the_index_sort_is_refused`,
    which patches a sort onto one of three otherwise identical segments -- so
    the sort declaration is the only difference -- and asserts both that the
    segment is still offered to the merge policy and that the merge is refused.

18. **[CORRECTNESS -> fixed]** *A sorted index never auto-merged.* c17's
    finding 13, which was containment rather than a fix: `segment_stats`
    refused to offer a segment whose `.si` declares a sort. Dropped, along
    with the `.dvd` exclusion (finding 15). **What is kept**: a segment with a
    doc-values *generation* (`doc_values_gen != -1`) is still withheld, because
    its newest column lives in generational files no `.si` lists and
    `execute_merge` does not open -- merging it would resurrect the base
    column silently. That is now the *only* thing withheld, which is what
    makes findings 22, 23 and 24 the price of this one: every format
    `execute_merge` does not open is a format the merge drops, and nothing
    downstream reports it.

### Verdict

Swept clean for the merge path. The one remaining un-mergeable case is named
and is a data-loss guard, not a scope gap. The lesson from findings 22-24 is
written into `describe_written_files`' doc comment rather than left here:
dropping an exclusion from `segment_stats` moves the burden onto
`execute_merge` to open *every* format the flush can write, and there is no
mechanism that notices when it does not.

---

## crates/lucene-index/src/merge_policy.rs

Java counterpart: `index/TieredMergePolicy.java`.

**No change, and that is the finding.**

20. **[INTENTIONAL]** *Nothing about an index sort belongs here.* Java's
    `TieredMergePolicy` never consults `segmentInfo.getIndexSort()` -- the
    decision of *which* segments to merge is size-and-deletes only, and the
    sort is `MergeState`'s business. Re-checked against the pinned 10.5.0
    source (`grep indexSort` over `TieredMergePolicy.java` and
    `MergePolicy.java`: no hits). This port matches. The change this batch
    makes that reaches the merge policy is `segment_stats` handing it more
    candidates (finding 18), which is a caller change; the scoring, the tier
    walk, `maxMergeAtOnce` (c18's finding 1) and `find_forced_delete_merges`
    are untouched.

### Verdict

Swept clean; no index-sort work belongs here.

---

## crates/lucene-codecs/src/vectors.rs (merge entry point only)

Java counterpart: `codecs/lucene99/Lucene99FlatVectorsWriter.mergeOneFlatVectorField`
plus the `MergedVectorValues.merge*VectorValues` iteration it consumes.

Additive change, scoped to `merge_one_flat_vector_field`; c21 owns `hnsw.rs`
and `hnsw_vectors.rs`, which are called and not edited.

19. **[CORRECTNESS -> fixed]** *Merged vector ordinals were assigned in source
    order.* `.vemf`'s doc list is an `IndexedDISI`, which encodes a **strictly
    ascending** list and nothing else. With no index sort, source order and
    merged-document order coincide; with one they do not, and the merge would
    have been rejected outright by `validate_docs` -- loudly, but it would have
    made the sorted merge impossible rather than wrong. Java assigns ordinals
    through `DocIDMerger.of(subs, mergeState.needsIndexSort)`, i.e. in merged
    document order. Now so does this: a plan of `(new_doc, source, ord)` is
    built before a byte is copied, sorted only when the scan finds it not
    already ascending (so the unsorted path moves nothing), and emitted with
    every run of consecutive ordinals from one source coalesced into a single
    `memcpy` -- the fast path is kept, not traded.
    A source whose *surviving* documents do not map to strictly increasing
    merged ids is now rejected per source with its own message. That is not
    pedantry: a `MergeState.DocMap` never reorders within a source (a merge
    drops documents and interleaves sources; even a sorted merge keeps each
    source's own order, because every source is already sorted by the same
    key), so a non-monotone one is a caller defect -- and it is exactly the
    defect the new interleaving sort would otherwise launder into a
    well-formed segment.
    *Tests*: `an_interleaving_doc_map_assigns_merged_ordinals_in_document_order`
    (checks each merged ordinal's doc id *and* its vector components against
    the document it came from) and the pre-existing
    `merge_rejects_a_doc_map_that_is_not_ascending_or_is_short`, which now
    fails on the per-source check rather than on `validate_docs`.

### Verdict

Swept clean for the merge entry point.

---

## Measurements

### What the sort costs a merge

`benchmarks/rust-runner/src/merge_bench.rs` gained a `sorted_merge_scenario`:
the **same sources**, merged twice, once with `sort_fields: None` and once
with a one-tier sort, so the delta is the sort and nothing else. Each source is
internally sorted by its own key and the sources' key ranges **fully overlap**
-- the worst case for run detection and the normal case for an index-sorted
index, where every flush covers the whole key range. 4 segments x 20 000
documents (two stored string fields, 4.9 MB of `.fdt`), best of three after a
warm-up, `--release`.

The scenario merges `MergeSource::stored_only` sources, so these are the
**stored-fields** merge's numbers, not a whole segment's: they isolate the one
path a sort actually changes the cost of.

| | before finding 9 | after finding 9 |
|---|---|---|
| concatenated (BULK) | 0.9 ms | 0.6-1.5 ms |
| index-sorted | **2 004.4 ms** | **10.9-23.2 ms** |
| ratio | 2 227x | **13-18x** |

Six runs across the session, three of them while other batches were building
the workspace; the ranges are that noise, and the quietest run is the
rightmost figure in each row (0.6 ms / 10.9 ms / 18.2x). The *ratio* moves
more than either arm does, because the concatenated arm is a `memcpy` and
gains more from an idle machine than the per-document arm does -- so the
honest statement is "an index-sorted merge of these sources costs somewhere
between 13x and 18x the concatenated one", not a single number. Both arms
merge the same sources in the same process, back to back.

The 152x between the columns is finding 9: one whole-chunk decompression per
document, because `copy_chunks` allocated a fresh cursor per call and a sorted
merge's runs are one document long.

The remaining ~15x is the honest cost and is **the byte-copy path becoming
illegal**, not a regression. c4 measured stored-fields BULK at 520x over the
pre-c4 merge and DOC at 26x; a sorted merge cannot use BULK, because a
compressed chunk encodes a contiguous run of documents and an index sort
interleaves the sources, so it lands on DOC. 13.2 ms for 80 000 documents is
6.1 M docs/s, which is c4's DOC figure. Java behaves identically: its
`getMergeStrategy` also returns BULK and lets `DocIDMerger`'s sorted mode
collapse the runs into `copyOneDoc` calls.

**What it is not**: a reason to avoid sorting. 10.9 ms to merge 80 000
documents is 7.3 M docs/s; the sort's cost lands entirely on a path that was
already three orders of magnitude faster than the pre-`c4` merge.

The strategy selection is therefore unchanged and correct-by-construction:
`write_merged_stored_fields` and `write_merged_term_vectors` detect runs from
`doc_order` itself, so BULK is taken exactly when it is legal, per source, per
run -- including on a sorted merge, where two adjacent documents from the same
source still form a run.

Every other figure in `merge-bench` is unchanged by this batch (stored fields
520x/26x/29x, postings 10.8x, BKD 2.6x, term vectors 758 205x BULK /
3 078x per-doc), re-run in the same session.

---

## Verification

**Rust writes, real Lucene reads** (new; `verify-write-path.sh` **20/20 ->
21/21**): `crates/lucene-index/examples/write_sorted_merged_segment_fixture.rs`
-> `fixtures/src/VerifySortedSegment.java`.

The verifier is **the same class** that checks the flushed sorted segment, and
that is the point: the claim under test is that a merged sorted segment is
*indistinguishable* from a flushed one. The fixture writes c17's exact corpus
-- 2 000 documents, a reversed first tier whose missing documents therefore
belong at the front, a second tier breaking its many ties, stored fields,
postings with a term unique to each document, norms encoding each document's
length, two NUMERIC columns (one sparse), and a FLOAT32 vector field past
`HNSW_GRAPH_THRESHOLD` -- but produces it from **eight overlapping flushes with
one document in fifty-three deleted**, merged into one segment.

That makes the merged case strictly harder than the flushed one:

- the eight sources' key ranges overlap completely, so a concatenation cannot
  come out ordered by accident;
- the deletions rule out both byte-copy paths and make every doc map
  non-trivial rather than a pure interleaving;
- one HNSW graph is rebuilt over a brand-new merged ordinal space;
- postings, doc values, norms, term vectors and vectors are each mapped
  through their own doc map.

Java then checks: `LeafMetaData.sort()` tier for tier including each tier's
`reverse` and `missingValue`; the physical order against a permutation it
re-derives itself from the fixture's generator functions with its own
comparator over the survivors; per doc id the stored `id`, both doc-values
columns (including `rank`'s *absence* where the document has none), the unique
postings term resolving to exactly that doc id, the norm, and every component
of the vector; that no document is still marked deleted; that a deleted
document's unique term is **gone from the merged dictionary** rather than
merely unreachable; a real `KnnFloatVectorQuery` over the merged graph; and
`CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS`, which runs Lucene's own
`testSort`.

`scripts/verify-write-path.sh` grew an optional fifth field per case so the
verifier can be told the fixture's deletion rule.

**Negative controls, in c17's shape.** c17 proved its work by showing that
dropping one format's permutation attaches every vector to the wrong document
*and still passes `CheckIndex`*. The equivalents here are tests, not by-hand
experiments:

- `a_concatenating_merge_of_the_same_segments_loses_the_order_silently` takes
  the three sorted segments, **strips the sort from their `.si` files** so
  `execute_merge` takes the concatenating path over the very same bytes, and
  asserts the merged segment is (a) in source-concatenated order, (b) still
  entirely self-consistent -- every format's per-document data checked
  individually -- and (c) **clean under this port's `CheckIndex`**, because it
  makes no claim about its order. Only comparing it against the sorted result
  shows the loss. Then the sort is stamped onto the merged `.si`, which is what
  a merge that kept the declaration while concatenating would have produced,
  and the order check fails. The concatenated order is worth stating: each
  source is internally sorted, so it is sorted *within* each source and jumps
  back down at every boundary -- it looks almost right.
- `the_doc_id_maps_invert_the_doc_order_for_both_merge_orders` pins the one
  invariant that makes findings 3, 4 and 19 hold: `doc_order` and
  `build_doc_id_maps` are inverses, and the map is increasing within a source.
  A one-document disagreement between them is exactly the "well-formed,
  checksummed, `CheckIndex`-clean segment whose contents are attached to the
  wrong documents" failure.
- `an_interleaving_doc_map_assigns_merged_ordinals_in_document_order` checks
  each merged vector ordinal's doc id *and* its components against the source
  document it came from, which is the assertion a wrong ordinal assignment
  breaks while leaving a segment that decodes cleanly.
- `take_permuted_moves_each_element_to_its_named_position` uses a 3-cycle,
  because an involution cannot distinguish a permutation from its inverse --
  the precise trap c17 recorded for `permute_in_place`.

**Multi-field, reverse, and deletions**, since a single ascending key with no
deletions is the case most likely to pass by accident:
`a_merge_honours_every_tier_of_a_multi_field_reverse_sort` (rank descending,
tie ascending, with both rank groups spanning all three sources so the second
tier does the ordering) and
`a_sorted_merge_drops_deleted_documents_from_every_format` (one deletion per
source, asserting the survivors' order, that `del_count` is zero afterwards,
and that the deleted documents' terms are gone from the merged dictionary).

**The shared assertion.** All four end-to-end tests go through
`assert_every_format_agrees`, which reads a segment back and checks that
stored fields, the doc-values column, the unique postings term, the norm, the
term vector, the vector's components and the merged graph's size **all
describe the same document at the same doc id**. Each of those is a different
file, so a doc map applied to one and not another shows up as a specific
mismatch rather than as "something is off".

**Unit tests**: 9 new and 2 rewritten in `index_writer.rs`; 4 new and 7
rewritten in `merge.rs`; 1 new in `codecs/vectors.rs`. Five of the new ones
are the Tier-2 review's findings 21, 22, 23, 24 and 26; the first two were
**verified to fail** with their fix stubbed back out, which is the only way to
know an assertion about a silent loss is load-bearing. One more is a
cross-module consistency check rather than a merge test: `a_doc_values_update_still_works_against_a_merged_segment` updates a
doc-values field on a segment produced by a **merge**, because the merged
`.fnm` is now written by `describe_written_files`' zeroing rule (finding 8) and
`field_updates::check_updatable` has to accept what it produces -- the two are
written in different modules and used to disagree (c17 findings 14/15).

**Gates**: `cargo fmt --all`, `cargo clippy -p lucene-index -p lucene-codecs
--all-targets -- -D warnings`, `cargo test -p lucene-index -p lucene-codecs`,
`python3 scripts/check-parity.py`, and `scripts/verify-write-path.sh` (22/22,
including both `VerifySortedSegment` cases) -- all green. Three other batches
were editing this workspace throughout; each of their transient breakages was
waited out and re-run rather than worked around, and the only file outside
this batch's own that it touched for them is one `as usize` clippy fix in
`index_writer.rs`. Coverage (`cargo llvm-cov --summary-only` with its own
`CARGO_TARGET_DIR`, per c10's note that a figure taken while anything else is
building the workspace is not a figure), lines: `merge.rs` **98.50%**,
`merge_policy.rs` **99.06%**, `index_writer.rs` **98.25%** -- all above the
95% bar, and `lucene-index` as a whole at 98.14%.

---

## Carry-over

- [ ] **Multi-field norms in a merge** (finding 13). Needs a
      `norms::write_dense_fields` the way doc values got one; the merge side is
      then the same three-line widening finding 5 was. Owner: a
      `lucene-codecs`/norms batch.
- [ ] **Generational doc values are still un-mergeable** (finding 18). A
      segment with `doc_values_gen != -1` is withheld from the merge policy
      because `execute_merge` opens only the base `.dvm`/`.dvd`. Closing it is
      "open the newest generation per field instead of the base one", which is
      `field_updates`/`c14` territory.
- [ ] **Points are still not reachable from a flush**, so `merge_points` runs
      only against hand-built `MergeSource`s. Unchanged from b10/c4; the merge
      side is now sort-correct (finding 3) and needs nothing when a flush path
      arrives.
- [ ] **`MergeSource` cannot carry per-source `min_version`** (b10), so a
      merged segment's `.si` reports this writer's version rather than the
      oldest source's. `has_blocks` is no longer in this bucket -- finding 24
      routes it through `MergeOptions` -- and the note that used to justify
      leaving it there ("not expressible anyway") was wrong for an unsorted
      merge.
- [ ] **`ConcurrentHnswMerger`** (c10 finding 35): the graph merge is
      single-threaded. Now that it is reachable from `execute_merge` it is the
      merge's dominant cost for a vector-heavy segment, so this moved from
      theoretical to measurable.
- [ ] **The index sort is still unreachable from the FFI boundary** (c17's
      carry-over, unchanged): `lucene-ffi` exposes no `set_index_sort`, so from
      OpenSearch neither the sorted flush nor the sorted merge can be turned
      on.
- [ ] **Nothing mechanically checks that `execute_merge` opens every format
      the flush can write.** Findings 22, 23 and 24 are three instances of one
      omission, and each was found by reading rather than by a gate: a format
      the merge does not open is a format it drops, and the merged segment is
      valid either way. The doc-values half now has a `debug_assert!` (every
      entry a source's `.dvm` declares reaches the merge); postings, term
      vectors, norms and vectors do not, and the general form -- "the merged
      `FieldInfos` must not lose a capability any source declared, unless the
      loss was named" -- would be a real check to build, most naturally in
      `check_index` against the source and merged `.fnm`s.
