# c17-index-sort

Follow-up batch opened from `c10-vectors-wiring`'s carry-over list: **make the
second correct-but-unreachable subsystem reachable.**

Several batches had built parts of index sorting and none of it was reachable
end to end. `b11` replaced the `.si` index-sort encoding (which had been this
port's own invention) with the real `SortFieldProvider` layout, proven both
directions against Lucene 10.5.0. `segment_writer.rs` had
`flush_sorted_stored_only_segment`; `merge.rs` had
`merge_sorted_stored_only_segments`. But `IndexWriter` invoked neither, there
was no `IndexWriterConfig.setIndexSort` equivalent, and so **no format's
index-sort write path could be taken** -- c10's finding 11.

Java counterparts (Lucene 10.5.0 at `/home/tuong/work/lucene-10.5.0`; the
`/home/tuong/work/lucene` checkout is `main`, 4574 commits ahead, and both
`IndexWriter` and `SegmentInfos` differ between the two -- every citation below
was re-checked against the pinned tree):

- `index/IndexWriterConfig.java` (`setIndexSort`, `indexSortFields`)
- `index/IndexingChain.java` (`maybeSortSegment`, `validateIndexSortDVType`,
  the per-format `writeNorms`/`writeDocValues`/`writePoints`/
  `vectorValuesConsumer.flush`/`storedFieldsConsumer.flush`/`termsHash.flush`
  `sortMap` fan-out)
- `index/IndexWriter.java` (`validateIndexSort`, `isCongruentSort`,
  `publishFlushedSegment`'s `rld.sortMap`, `updateNumericDocValue`/
  `updateDocValues`' index-sort-field guard)
- `index/Sorter.java` (`DocMap`, `sortAndLeaveUnpacked`),
  `index/IndexSorter.java` (`LongSorter.getDocComparator`)
- `index/FrozenBufferedUpdates.java` (the two `sortMap.newToOld(doc) < limit`
  branches), `index/ReadersAndUpdates.java` (`sortMap`)
- `index/CheckIndex.java` (`testSort`)
- `index/DocumentsWriterPerThread.java`, `index/SortingStoredFieldsConsumer.java`,
  `index/SortingTermVectorsConsumer.java`, `index/SortedDocValuesWriter.java`,
  `codecs/lucene99/Lucene99FlatVectorsWriter.java` (`writeSortingField`)

Totals: **19 findings** -- 4 CORRECTNESS (all fixed), 10 MISSING (9 fixed, 1
recorded with a named blocker), 1 PERF (measured), 4 INTENTIONAL. One of the
CORRECTNESS findings (11b) came from the Tier-2 review and was a defect this
batch introduced.

`scripts/verify-write-path.sh`: **19/19 -> 20/20** (confirmed by running it,
not assumed -- `c10` recorded 19 and a concurrent batch's `dv-updates` case had
since made it 19; this batch adds the 20th).

---

## crates/lucene-index/src/segment_writer.rs

Java counterparts: `index/IndexSorter.java` (`LongSorter.getDocComparator`),
`index/Sorter.java` (`DocMap`, `sortAndLeaveUnpacked`),
`index/DocumentsWriterPerThread.flush`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `sort_key_rank` | `IndexSorter.LongSorter.getDocComparator`'s `reverseMul * Long.compare(values[d1], values[d2])` over an array pre-filled with `missingValue` | **was divergent (finding 1)**; now identical |
| `sort_permutation` (new) | `Sorter.sortAndLeaveUnpacked` | equivalent: stable, priority-ordered, doc id as the final tie-break |
| `permute_in_place` (new) | `Sorter.DocMap` applied by each format's writer | not-in-Java by design (finding 5); one buffer permutation instead of N per-format remaps |
| `flush_sorted_stored_only_segment` | `DocumentsWriterPerThread` sort-on-flush, stored-fields half | unchanged in behaviour; now shares `sort_permutation` with the writer so the two orders cannot drift |

Java with **no** Rust counterpart: the per-format `Sorter.DocMap` consumers
(`SortingStoredFieldsConsumer`, `SortingTermVectorsConsumer`,
`{Numeric,Sorted,Binary,SortedNumeric,SortedSet}DocValuesWriter.sortDocValues`,
`PointValuesWriter.MutableSortingPointValues`,
`Lucene99FlatVectorsWriter.writeSortingField`) -- deliberately, see finding 5.

### Findings

1. **[CORRECTNESS -> fixed]** *The missing-value comparator disagreed with the
   `.si` it wrote.* `segment_info::write_sort_field` emits
   `SortField(field, Type.LONG, reverse)` with an explicit `missingValue` of
   `Long.MIN_VALUE` (`SortMissingValue::First`) or `Long.MAX_VALUE`
   (`Last`). Lucene's reader-side comparator for that is

   ```java
   long[] values = new long[maxDoc];
   Arrays.fill(values, missingValue);
   ... values[docID] = dvs.longValue(); ...
   return (d1, d2) -> reverseMul * Long.compare(values[d1], values[d2]);
   ```

   -- the sentinel is an ordinary value inside the comparison, so **`reverseMul`
   applies to it too**: with `reverse`, a `Last` document sorts *first*.
   `sort_key_rank` instead bucketed missing values to one end **regardless of
   `reverse`**, and both the module doc comment on `SortMissingValue` and a
   dedicated test (`..._placement_is_independent_of_reverse`) asserted that
   wrong behaviour. Consequence: every segment this port wrote with a reversed
   sort *and* any missing key was physically ordered one way and described the
   other way by its own `.si`. Nothing local catches it -- the files are all
   valid.
   **Measured, both directions.** Read direction: the new
   `fixtures/data/sorted_index/` was written by a real `IndexWriter` with
   `rank` descending / missing-last, and Lucene put its six missing-`rank`
   documents **first**; with the old comparator
   `check_index`'s `sort.docs_in_index_sort_order` rejects that real index
   (`docID=5 sorts after docID=6`). Write direction: with the old comparator,
   real `CheckIndex` rejects the segment this port writes --
   `CheckIndexException: segment has indexSort=<long: "rank">! ... but
   docID=1944 sorts after docID=1945`.
   *Tests*: `sorted_flush_missing_value_sentinel_is_reversed_like_any_other_value`
   (the rewritten unit test), `sort_permutation_is_priority_ordered_stable_and_sentinel_reversing`,
   `a_missing_sort_key_takes_its_sentinel_and_reverses_with_it` (both
   directions of `reverse`, through the writer and this port's `CheckIndex`),
   `tests/index_sort_fixtures.rs::our_own_sort_check_accepts_a_real_lucene_sorted_segment`,
   and `VerifySortedSegment`.

2. **[CORRECTNESS -> fixed]** *`permute_in_place` applied the inverse
   permutation.* Caught during development, recorded because the failure mode
   is instructive: the classic in-place cycle walk implements `oldToNew`, and
   feeding it a `newToOld` map produces a permutation that is total,
   self-consistent, deterministic and wrong -- and *indistinguishable* on any
   involution, so a two-element or fully-reversed test cannot tell the two
   apart. `permute_in_place` now inverts explicitly and the test uses a
   3-cycle.
   *Test*: `permute_in_place_applies_new_to_old_not_its_inverse`.

3. **[MISSING -> fixed]** *No shared permutation.* `sort_permutation` and
   `permute_in_place` are new, and `flush_sorted_stored_only_segment` was
   rewritten onto the first of them, so the order this module's own primitive
   imposes and the order `IndexWriter::flush` imposes are one function.

4. **[PERF]** *`permute_in_place` is a cycle walk, not a permuted copy.* The
   buffers being permuted hold whole `Document`s and per-document vector
   lists; building a permuted copy would hold two full buffers live at once,
   doubling the peak footprint of exactly the structure a flush exists to
   discharge. Cycle-following moves each element at most twice and allocates
   one `u32` per document. See Measurements.

5. **[INTENTIONAL]** *One buffer permutation instead of Java's per-format
   `Sorter.DocMap`.* Java hands a `DocMap` to **every** format's writer and
   each remaps its own doc ids; there are eight such consumers and adding a
   format means remembering to add a ninth. This port permutes
   `pending_docs` (and the two buffers aligned with it) once, before any
   format is built, because every consumer in `IndexWriter::flush` already
   derives a document's id from its **index** in that buffer. The result is
   the same physical layout with the "did I remember this format?" question
   removed -- which is not hypothetical: c10 found Lucene's vector
   `writeSortingField` unreachable here for exactly that reason, and the
   negative control below shows that forgetting one format is
   `CheckIndex`-clean.

### Verdict

Swept clean. The comparator now means what the bytes say, and the permutation
is single-sourced.

---

## crates/lucene-index/src/index_writer.rs

Java counterparts: `index/IndexWriterConfig.setIndexSort`,
`index/IndexWriter.{validateIndexSort,isCongruentSort,publishFlushedSegment,
updateNumericDocValue,updateDocValues}`, `index/IndexingChain.{maybeSortSegment,
validateIndexSortDVType,flush}`, `index/FrozenBufferedUpdates`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `set_index_sort` | `IndexWriterConfig.setIndexSort` + `IndexingChain.validateIndexSortDVType` + `IndexWriter.validateIndexSort` | **added** (findings 6, 7, 8, 9, 10); the three checks Java splits across construction, first-document and per-field are all done here, because this port's field list and segment list are both known at configuration time |
| `index_sort` | `IndexWriterConfig.getIndexSort` | added |
| `validate_index_sort_against_existing_segments` | `IndexWriter.validateIndexSort` + `isCongruentSort` | identical rule (prefix), including that a segment with **no** sort fails |
| `sort_pending_buffer` | `IndexingChain.maybeSortSegment` + the eight per-format `sortMap` consumers | **added** (finding 6), one permutation (finding 5) |
| `write_index_sort_to_si` | `SegmentInfo.setIndexSort` before `Lucene99SegmentInfoFormat.write` | added; patches the `.si` last, the established pattern here |
| `apply_packets_to_segment`'s `below_limit` | `FrozenBufferedUpdates`' two `segState.rld.sortMap.newToOld(doc) < limit` branches | **added** (finding 11) |
| `add_doc_values_field` | `IndexingChain`'s per-field `DocValuesWriter` (one per DV field, always) | **added** (finding 12) |
| `build_doc_values_output` / `collect_dense_column` / `collect_*_values` | `IndexingChain.writeDocValues` + `Lucene90DocValuesConsumer` | widened to a field list; the five per-type extractors are now shared between the single-field and multi-field paths |
| `segment_stats` | `IndexWriter.updatePendingMerges` -> `MergePolicy.findMerges` | index-sorted segments excluded (finding 13) |
| `verify_doc_values_update_field` | `IndexWriter.updateDocValues`' `config.getIndexSortFields().contains(field)` guard | **added** (finding 9) |

Java with **no** Rust counterpart: the parent-field path
(`maybeSortSegment`'s `BitSet parents` comparator wrapper) -- this port has no
parent-field write path and refuses the combination instead (finding 8);
`Sorter.PackableDocMap`'s packing (this port keeps the plain `Vec<usize>`,
which is what its scoped lifetime makes affordable).

### Findings

6. **[MISSING -> fixed]** *`IndexWriter` had no index sort at all.* c10's
   finding 11. `set_index_sort(Option<&[IndexSortField]>)` /
   `index_sort()` are the configuration surface, and `flush` now calls
   `sort_pending_buffer` before building any format. The permutation is
   derived from each sort field's NUMERIC value **read out of the same
   `Document`s `build_doc_values_output` writes the column from**, so the
   order and the column a reader checks it against cannot come from two
   different facts. `Sorter.sortAndLeaveUnpacked`'s "already sorted -> no
   `DocMap`" case is reproduced (the sort is still recorded in the `.si`; there
   is simply nothing to permute).
   *Tests*: `a_sorted_flush_reorders_every_format_together`,
   `a_multi_field_sort_breaks_ties_with_its_second_tier`,
   `an_already_ordered_buffer_is_still_recorded_as_sorted`,
   `every_sorted_flush_gets_its_own_ordered_segment`, plus the whole
   `VerifySortedSegment` case.

7. **[MISSING -> fixed]** *`setIndexSort`'s validation.* Four rules, three of
   them Java's and one this port needs because its doc values are opt-in:
   - non-empty (`numSortFields == 0` *is* unsorted on disk, so an empty sort
     has no encoding distinct from `None`);
   - every field is NUMERIC doc values --
     `IndexingChain.validateIndexSortDVType`, narrowed to the single
     `SortField.Type` this port's `.si` encoder emits;
   - every field is congruent with every segment already in the index --
     `IndexWriter.validateIndexSort` + `isCongruentSort` (prefix), checked
     against the last commit's segments *and* the flushed-but-uncommitted
     ones by reading each `.si`;
   - every field is actually **opted into doc values**. Java gets this free
     (a field with a `DocValuesType` always gets a `DocValuesWriter`); here it
     matters because `DocValues.getNumeric` substitutes an *all-missing*
     column rather than failing, so a sort over an unwritten column makes
     `CheckIndex.testSort` compare `maxDoc` equal keys and pass. A sort no
     reader can evaluate is worse than no sort.
   *Tests*: `set_index_sort_rejects_an_empty_sort`,
   `set_index_sort_rejects_an_unknown_field`,
   `set_index_sort_rejects_a_field_that_is_not_numeric_doc_values`,
   `set_index_sort_rejects_a_sort_field_no_doc_values_are_written_for`,
   `set_index_sort_must_be_congruent_with_the_segments_already_in_the_index`
   (prefix accepted, identical accepted, opposite direction rejected, longer
   rejected, unsorted existing segment rejected).

8. **[MISSING -> fixed]** *Document blocks plus an index sort.* A block added
   by `add_documents`/`update_documents` must stay physically contiguous and
   in order; a sort would shred it. Java allows the combination **only** when
   a parent field marks each block's last document, and otherwise raises
   `CorruptIndexException("parent field is not set but the index has blocks
   and uses index sorting")` for any index created at 10.0 or later. This port
   has no parent-field write path, so the flush refuses rather than producing
   a segment whose blocks are silently interleaved.
   *Test*: `document_blocks_and_an_index_sort_are_refused_together`.

9. **[MISSING -> fixed]** *A doc-values update against a sort field.* Java:
   `IllegalArgumentException("cannot update docvalues field involved in the
   index sort")`, in both `updateNumericDocValue` and `updateDocValues`.
   Rewriting the column the segment's physical order is defined over leaves
   every existing segment claiming a sort it no longer satisfies, silently --
   nothing re-checks the order after an update.
   *Test*: `a_doc_values_update_on_a_sort_field_is_refused` (and the control:
   a doc-values field that is *not* in the sort still updates).

9b. **[MISSING -> fixed]** *Narrowing the doc-values field list could strand
    a configured sort.* `set_doc_values_field(Some(x))` **replaces** the list,
    so calling it after `set_index_sort` could leave a tier with no column --
    the exact state finding 7 refuses to create, reached from the other side.
    Java cannot reach it (immutable config, doc values not opt-in). Checked
    before the mutation, so a rejected call leaves the list untouched.
    *Test*: `narrowing_the_doc_values_fields_cannot_strand_a_configured_index_sort`,
    including the two legal cases (clear the sort first; narrow to exactly a
    single tier's own column).

10. **[INTENTIONAL]** *`set_index_sort` is refused once documents are
    buffered.* Java cannot reach the state at all -- `IndexWriterConfig` is
    snapshotted when the `IndexWriter` is constructed. This writer's opt-ins
    are read at flush (c10 finding 43), so changing the sort mid-buffer would
    order part of a batch by one key and the rest by another. Refusing is the
    faithful lowering of "the sort is fixed for the writer's life", not an
    extra rule; clearing the sort is refused on the same terms.
    *Test*: `set_index_sort_is_refused_once_documents_are_buffered`.

11. **[CORRECTNESS -> fixed]** *A segment-private delete's `docIDUpto` is a
    **pre-sort** position.* `update_document` buffers its delete at the buffer
    position it was issued at, and the private packet's limits are compared
    against doc ids in the flushed segment. Once the flush permutes the
    buffer those are different spaces, so the limit must be compared against
    `newToOld(doc)`. Java keeps a `Sorter.DocMap` on the pooled
    `ReadersAndUpdates` for exactly this, and applies it in two places
    (`applyQueryDeletes`, `applyDocValuesUpdates`); this port needs it in
    three, because Java resolves a private packet's *term* deletes before the
    sort while this port resolves them after. Without it a sorted flush
    silently deletes the wrong documents -- in the test's descending sort,
    exactly the complement of the right ones.
    `pending_sort_map` is scoped to the `flush()` call rather than pooled for
    the writer's lifetime: only the packet whose generation *equals* the
    segment's has a limit below `MAX_DOC_ID_UPTO`, and
    `apply_all_deletes_and_updates` drains the whole stream inside that call.
    The query-delete path additionally drops its `query_bound(limit)` early
    exit for a sorted segment, because the documents below the limit are no
    longer a prefix -- Java's sorted branch drops the same loop bound.
    *Test*: `a_private_delete_limit_is_mapped_back_through_the_sort`.

11b. **[CORRECTNESS -> fixed, from the Tier-2 review]** *A flush that failed
    **after** the sort left the buffer permuted, and the retry then mis-mapped
    the delete limits.* `sort_pending_buffer` mutates the buffer, and five
    fallible build steps followed it with a bare `?` straight out of `flush`,
    each returning before `pending_sort_map` was assigned. The writer keeps
    its documents on a failure (so the caller can repair and retry, which
    `set_doc_values_field`'s own doc comment promises), so the retry saw an
    **already-ordered** buffer, took `sort_permutation`'s identity
    short-circuit, produced no map -- and then compared the private packet's
    pre-sort `docIDUpto` against sorted doc ids. Finding 11's defect,
    reintroduced through the error path, and just as silent: same `del_count`,
    valid `.liv`, clean `CheckIndex`. With the descending sort of finding 11's
    own test it deletes `{p3,p2,p1}` where it must delete `{p2,p1,p0}` -- the
    exact complement.
    Fixed by making the whole build-and-write region one method
    (`build_and_write_segment`, `&self`, so nothing in it can mutate the
    buffer) with **one** error path, which restores insertion order via
    `unsort_pending_buffer` before returning. Insertion order is then a
    standing invariant of the buffer, which is what every `docIDUpto` a
    delete records means.
    *Test*: `a_flush_that_fails_after_the_sort_leaves_the_buffer_in_insertion_order`,
    which injects the failure the way a caller would hit it (a non-numeric
    value in a second doc-values field), asserts the buffer's order directly
    -- the invariant, not only its consequence -- then repairs the document,
    retries, and asserts the surviving document set. Verified to fail without
    the restore.

12. **[MISSING -> fixed]** *One doc-values field per segment.* `IndexingChain`
    creates a `DocValuesWriter` for **every** field whose `FieldType` declares
    a type; this writer had a single `Option<DocValuesFieldConfig>` and no
    `add_doc_values_field`, which made a **multi-field index sort
    inexpressible** -- a second tier is a second doc-values column.
    `doc_values_fields` is now a list, written into one `.dvm`/`.dvd`/`.dvs`
    triple through `doc_values::write_dense_fields`, which is what a real
    multi-field `Lucene90DocValuesFormat` segment looks like.
    *Tests*: `two_doc_values_fields_share_one_dvm`,
    `a_multi_field_doc_values_flush_allows_a_sparse_numeric_but_not_a_sparse_sorted`,
    `a_multi_field_sort_breaks_ties_with_its_second_tier`, and the
    `VerifySortedSegment` case (two columns read back through real Lucene).

13. **[CORRECTNESS -> fixed]** *An automatic merge would have desorted the
    index.* `execute_merge` goes through `merge::merge_stored_only_segments`,
    which concatenates its sources in source order and writes `index_sort:
    None` -- correct for what it produced, but it turns a sorted index into an
    unsorted one with nothing reporting it (the merged segment is valid,
    `CheckIndex` is clean, and the only visible consequence is that a later
    `set_index_sort` fails congruence). `segment_stats` now refuses to offer a
    segment whose `.si` declares a sort to the merge policy, which is the same
    "un-mergeable beats mergeable-with-silent-loss" rule that file already
    applies to doc-values-bearing segments. In practice the `.dvd` rule
    already covered every sorted segment (finding 7 makes doc values
    mandatory for a sort field); the rule is now stated where it belongs
    rather than left as a consequence of another one.
    *Test*: `a_segment_whose_si_declares_an_index_sort_is_never_offered_to_the_merge_policy`,
    which patches a sort onto the `.si` of a segment with **no** doc values,
    so the pre-existing exclusion cannot be what does the work.

14. **[MISSING, recorded]** *A `.fnm` claiming doc values the segment does not
    carry.* The doc-values twin of c10's finding 2 (vectors) and of the norms
    rule already in `fields_with_per_field_attributes`: this port's field list
    is fixed at `open`, so the `.fnm` declares a `DocValuesType` for every
    field the caller listed, whether or not the flush wrote a column for it.
    Java never reaches the state (`IndexingChain` creates the field from the
    first document that carries it, so `.fnm` and `.dvm` come from one fact).
    Real `CheckIndex.testDocValues` iterates every field whose `.fnm` claims
    doc values and dereferences the producer `PerFieldDocValuesFormat` did not
    register; this port's own `check_index` reports
    `doc_values.entry_present:<field>`.
    **Attempted and reverted within this batch**, which is why it is recorded
    rather than fixed: zeroing the type broke every doc-values *update*
    against a field the base flush wrote no column for, because
    `field_updates::check_updatable` required the base `.fnm` to already
    declare the type. Half of that is now fixed (finding 15); the other half
    -- zeroing the claim -- needs a sweep of `write_full_segment_fixture`,
    `write_doc_values_updates_fixture` and `VerifyDocValuesUpdates` to confirm
    no committed fixture depends on the claim, which is `c14`'s territory and
    not this batch's. The sorted case specifically is *not* exposed: finding
    7 makes a sort field's column mandatory.

15. **[MISSING -> fixed]** *`verifyOrCreateDvOnlyField`'s create half.*
    `IndexWriter.updateDocValues` calls
    `globalFieldNumberMap.verifyOrCreateDvOnlyField(field, type, true)`, which
    **creates** the field as doc-values-only when it does not already carry
    doc values; the update's own generation is then its first column.
    `field_updates::check_updatable` rejected `DocValuesType::None` outright,
    so "this segment has not written a column for the field yet" was an error.
    It now accepts `None` and stamps the type into the generational `.fnm`
    (which `finish_generation` already stamps the `PerFieldDocValuesFormat`
    attributes onto -- c14 had handled the adjacent case).
    *Tests*: the existing `update_numeric_doc_value_writes_a_generation_the_reader_can_replay`
    and six other dv-update tests, which exercise the path through
    `sortable_fields`-shaped writers; `VerifyDocValuesUpdates` unchanged and
    green.

16. **[INTENTIONAL]** *`IndexWriter::flush` does not call
    `segment_writer::flush_sorted_stored_only_segment`.* c10's finding 11
    prescribed exactly that, and it is wrong: that function sorts and writes
    the **stored fields only**, while `IndexWriter::flush` builds postings,
    term vectors, norms, doc values and vectors from the *unsorted* buffer
    before and after it. Calling it would produce a segment whose stored
    fields are in sort order and whose every other format addresses the
    original doc ids -- corrupt, and `CheckIndex`-clean for the postings.
    The two now share `sort_permutation` instead, and
    `flush_sorted_stored_only_segment` remains the lower-level primitive it
    was, with its own tests.

### Verdict

Swept clean for the flush path. Open: the merge (finding 13's handoff below)
and finding 14.

---

## crates/lucene-index/src/indexing_chain.rs

Java counterpart: `index/IndexingChain.java`.

**No change, and that is the finding.**

17. **[INTENTIONAL]** *The sort does not belong in this module.* This port's
    `indexing_chain.rs` is Java's tokenize-and-invert half only: it takes
    `(doc_id, field, text)` triples and returns an inverted index. Java's
    `maybeSortSegment` runs in `IndexingChain.flush`, *before* the per-format
    consumers, and this port's equivalent point is `IndexWriter::flush` --
    which is where the buffer that defines doc ids lives. Routing the sort
    through this module would mean giving it the document buffer and the
    doc-values values it otherwise never sees, and it would still not reach
    stored fields or vectors, neither of which pass through it. Same
    conclusion c10 reached for vectors (its finding 9), for the same reason.

### Verdict

Swept clean; no index-sort work belongs here.

---

## crates/lucene-codecs/src/doc_values.rs (one variant)

Java counterpart: `codecs/lucene90/Lucene90DocValuesConsumer.addNumericField`.

`DenseField::SparseNumeric` added to the multi-field writer -- `(doc_id, value)`
pairs written through the same `IndexedDISI` + values body
`write_single_sparse_numeric_field` already used. Covered by finding 12: a
multi-tier index sort needs every tier's column in one `.dvm`/`.dvd`, and a
sort field whose documents may lack a value is the normal case
(`SortField.setMissingValue`), so a dense-only multi-field writer cannot
express the sorts Lucene routinely writes. The other four types stay
dense-only in the multi-field path and a sparse one is
`Error::SparseFieldInMultiFieldDocValues` naming the field and the count.

---

## Measurements

### What the sort costs per document

`benchmarks/rust-runner`'s `index-bench` gained `LUCENE_RUST_INDEX_SORT` (0/1/2
tiers) and `LUCENE_RUST_DOC_VALUES_ONLY` -- the latter is the honest control:
it writes the *same* two NUMERIC doc-values columns and does not sort, so the
delta is the sort rather than the columns it is defined over. 50 000 documents
x 40 tokens, postings + norms, one core, `--release`.

**Default 16 MB RAM buffer (several flushes), seven interleaved runs, µs/doc:**

| | runs | median |
|---|---|---|
| no doc values, no sort | 22.0 21.5 19.2 21.3 21.5 21.3 19.9 | **21.3** |
| two DV columns, no sort | 21.7 21.6 20.4 21.0 20.3 21.2 20.5 | **21.0** |
| two DV columns, 1-tier sort | 21.7 21.9 21.2 20.4 21.2 22.0 21.2 | **21.2** |
| two DV columns, 2-tier sort | 22.1 22.5 21.4 19.9 21.2 21.1 21.2 | **21.2** |

**The sort is free at this scale**, and the baseline lands exactly on c3/c7's
20-21 µs/doc.

**One flush of all 50 000 documents** (`LUCENE_RUST_RAM_BUFFER_MB=4096`, so the
permutation is over the whole corpus at once rather than per RAM-buffer batch),
five interleaved pairs:

| | runs | median |
|---|---|---|
| two DV columns, no sort | 23.6 22.8 24.4 24.4 23.1 | **23.6** |
| two DV columns, 2-tier sort | 24.6 23.9 28.7 23.8 22.9 | **23.9** |

**+0.3 µs/doc, ~1%**, which is the right order: a stable two-tier sort of 50k
`i64` keys is ~800k comparisons, and the three cycle permutations are O(n)
moves. Both arms flush one segment (asserted from the bench's own
`segments=` line).

The cost is bounded by the RAM buffer, not by the index: the permutation is
per flush, so `n` is however many documents one flush holds, and the sort is
`O(n log n)` in comparisons and `O(n)` in moves with one `u32` per document of
scratch (finding 4).

### Merge cost

Not measured: the sort-preserving merge is not wired (finding 13 / handoff).

---

## Verification

**Java writes, Rust reads** (new): `fixtures/src/GenSortedIndex.java` ->
`fixtures/data/sorted_index/` -> `crates/lucene-index/tests/index_sort_fixtures.rs`.
A real `IndexWriter` with `setIndexSort(new Sort(rank DESC missing=MAX, tie ASC
missing=MIN))`, 24 documents in two commits force-merged into one segment, with
duplicate ranks the second tier has to break and six documents missing `rank`
(and four missing `tie`). Four tests: the `.si` parses back to the configured
sort tier for tier and matches Lucene's own `Sort.toString()`; the documents
come back in Lucene's physical order; **this port's comparator reproduces that
order from Lucene's own columns** (through `check_index`'s
`sort.docs_in_index_sort_order`, i.e. the port of `testSort`); and the commit
still reads back. The third is the one that discriminates -- with the
pre-batch comparator it fails on the real index at `docID=5`.

**Rust writes, real Lucene reads** (new, `verify-write-path.sh` **19/19 ->
20/20**): `crates/lucene-index/examples/write_sorted_segment_fixture.rs` ->
`fixtures/src/VerifySortedSegment.java`. A 2000-document index written through
`IndexWriter` with the same awkward two-tier sort, carrying stored fields,
postings (a term unique to each document plus a shared one), norms (a length
that is a function of the document), two NUMERIC doc-values columns (one of
them sparse), and a FLOAT32 vector field past `HNSW_GRAPH_THRESHOLD`. Java
opens it with `DirectoryReader` and checks:

- `LeafMetaData.sort()` -- both tiers' field, type, `reverse` and
  `missingValue`;
- the physical order, against a permutation **Java re-derives itself** from the
  fixture's generator functions with its own comparator;
- per doc id: the stored `id`, both doc-values columns (including `rank`'s
  *absence* on the documents that have none), the unique postings term
  resolving to exactly that doc id, the norm, and every component of the
  vector;
- a real `KnnFloatVectorQuery` over the Rust-built graph, whose ordinals live
  in the sorted space, returning the probe document itself;
- `CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS`, which runs Lucene's own
  `testSort`.

**Two negative controls, run by hand**, both of which matter because they show
what the cheap checks do *not* catch:

- **dropping `permute_in_place(&mut self.pending_vectors, ...)`** (c10's
  hazard, raised by the coordinator mid-batch): six explicit mismatches
  (`vector on docID=1 component 0: 1.0 != 1369.0`) and a `KnnFloatVectorQuery`
  returning the wrong document -- and **`CheckIndex` stays clean**. Every
  vector attaches to the wrong document, silently. This port avoids it not by
  remembering to remap vector ordinals but by permuting the buffer the
  ordinals are assigned from (finding 5); the control proves the assertion is
  load-bearing either way.
- **restoring the pre-batch `sort_key_rank`**: real `CheckIndex` fails with
  `segment has indexSort=<long: "rank">! missingValue=9223372036854775807,...
  but docID=1944 sorts after docID=1945`.

**Unit tests**: 18 new in `index_writer.rs`, 2 new in `segment_writer.rs`, 1
rewritten there (finding 1's), 4 new integration tests. The load-bearing ones
are the ones that check *association* rather than validity:
`a_sorted_flush_reorders_every_format_together` walks stored fields,
doc values, postings, norms, term vectors and vectors and asserts each
describes the same document at the same doc id, and
`a_private_delete_limit_is_mapped_back_through_the_sort` asserts the exact
complement of documents survives.

**Gates**: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D
warnings`, `cargo test --workspace` (569 in `lucene-index`'s lib alone), and
`scripts/verify-write-path.sh` (20/20) -- all green. Coverage
(`cargo llvm-cov --summary-only` with its own `CARGO_TARGET_DIR`, per c10's
note that a figure taken while anything else is building the workspace is not
a figure), lines: `index_writer.rs` 98.10%, `segment_writer.rs` 99.49%,
`doc_values.rs` 98.27%, `field_updates.rs` 96.32%.

---

## Tier 2 semantic review

Run on the diff (`quality-reviewer`), against the pinned 10.5.0 tree. It
verified the four load-bearing claims independently:

- `sort_key_rank` reproduces `IndexSorter.LongSorter.getDocComparator`'s
  relation exactly, and `SortField.java`'s `Type.LONG` branch does pass the
  explicit `missingValue` through -- and noted that the one Java case not
  modelled (`missingValue == null`, where Java leaves the array zero-filled)
  is unreachable because `write_sort_field` always emits a sentinel and
  `parse` rejects the no-missing-value encoding;
- `pending_docs`/`pending_custom_freq_terms`/`pending_vectors` really are the
  complete set of per-document buffers, and every consumer in `flush` takes
  the buffer index as its doc id, so the one-permutation design is equivalent
  to Java's per-format `Sorter.DocMap`;
- `below_limit` is the only place any limit is compared anywhere in
  `lucene-index`;
- `validate_index_sort_against_existing_segments` is `isCongruentSort`.

**Three gating findings, all fixed:**

- finding 11b above -- the failed-flush path, which is the one real defect and
  is a defect this batch introduced;
- the `SortMissingValue` enum's own doc comment still said missing values sort
  "first or last regardless of ascending/descending" -- i.e. the thing finding
  1 corrected everywhere else. It is the doc every caller of `set_index_sort`
  reads to choose a variant, and it was self-contradicting (a sentinel *and*
  reverse-independent cannot both hold). Rewritten to name the sentinel, with
  the four-cell table;
- `set_doc_values_field`'s contract still said every field must be dense in a
  multi-field configuration, which finding 12 had just stopped being true --
  and the NUMERIC exception is the batch's headline enablement (a sort tier
  with missing values).

**Five advisories, four acted on:**

- `write_dense_fields`' `SparseNumeric` variant emitted an `IndexedDISI`
  region even when every document had a value, where
  `Lucene90DocValuesConsumer.writeValues` writes the `[-1, 0]` dense marker --
  unreachable from `IndexWriter` (`collect_dense_column` only produces the
  variant for a genuinely sparse column) but this is a public codec API, so
  the symmetric downgrade is now there;
- `below_limit` indexes the sort map directly; the length invariant is
  maintained by three separate assignments rather than by a type, and is now
  pinned with a `debug_assert!` where the map is bound;
- `GenSortedIndex`'s Javadoc said "two documents with no rank"; it is six
  (three rows x two batches), which is what the Rust test asserts;
- `VerifySortedSegment`'s norm comment claimed to compare "decoded-length
  classes" when it compares the raw byte (exact only because
  `SmallFloat.intToByte4` is the identity below 8), and `index_bench` had a
  stray single-statement block left over from an edit.

The fifth is recorded rather than acted on and is worth stating plainly,
because it is the batch's own argument turned back on it: the three parallel
per-document buffers are kept in step **by hand** at four push sites, four
clear sites and now three `permute_in_place` calls. The permutation design
exists so that no *format* can be forgotten, and it reintroduces the same
hazard one level down -- a fourth per-document buffer means remembering a
fourth `permute_in_place`. A `Vec<PendingDoc { doc, custom_freq_terms,
vectors }>` makes it structurally impossible and collapses the permutation to
one call. That is a refactor of `add_document`/`add_documents`/
`add_document_with_vectors`/`rollback`/`delete_all` as well, so it is on the
carry-over list rather than in this batch.

---

## Handoff: preserving the sort across a merge

Not done, and deliberately blocked rather than half-wired (finding 13). The
precise state:

1. `merge::merge_sorted_stored_only_segments` exists and does the k-way merge
   correctly for stored fields, all five doc-values kinds, norms and term
   vectors, and writes the shared sort into the merged `.si`. It writes **no
   postings, no points and no vectors** -- a documented scope limit from b10,
   also on `c4-merge-fastpath`'s carry-over list.
2. `IndexWriter::execute_merge` opens each source's postings and term vectors
   and goes through `merge::merge_stored_only_segments`. Routing it through
   the sorted entry point as things stand would trade a lost sort for lost
   postings, which is strictly worse.
3. So `IndexWriter::segment_stats` excludes any segment whose `.si` declares a
   sort, exactly as it already excludes doc-values-bearing ones. A sorted
   index therefore never auto-merges. That is a scope limitation, not a
   correctness hazard: every segment stays sorted and every `.si` stays true.

To close it, in order:

- give `merge_sorted_stored_only_segments` the postings merge
  `merge_stored_only_segments` already has (`merge_postings` is driven by
  `per_source_maps` + `per_source_live_ids` + `doc_order`, and the sorted
  entry point already computes all three -- the work is threading `doc_order`
  through `merge_postings`, whose current shape assumes source-concatenated
  order);
- add the vector half using c10's handoff verbatim (its five steps apply
  unchanged; the merged ordinal space is defined by
  `FlatVectorsWriter::merge_one_flat_vector_field` in whatever order
  `doc_order` produces);
- have `execute_merge` pick the sorted entry point when every source's `.si`
  declares the same sort, and refuse (rather than concatenate) when they
  disagree -- Java's `IllegalArgumentException("cannot change index sort
  from ... to ...")` at `IndexWriter.java:3107`;
- drop the `si.index_sort.is_some()` exclusion from `segment_stats`;
- extend `write_merged_segment_fixture`/`VerifyMergedSegment` with a sorted
  case: `VerifyMergedSegment` already opens the merged index with
  `DirectoryReader` and runs `CheckIndex`, so it needs only the
  `LeafMetaData.sort()` and per-doc-id assertions `VerifySortedSegment` now
  has.

## Carry-over

- [x] **Sort-preserving merge** -- the handoff above. **Closed by
      `c22-sorted-merge`**, though not by following the recipe: rather than
      giving `merge_sorted_stored_only_segments` the three missing formats,
      that batch collapsed both entry points into one `merge_segments` where
      `sort_fields` changes only the document order, so a format cannot be
      written by one and forgotten by the other. `segment_stats`' index-sort
      exclusion (finding 13) is gone, and `execute_merge` validates the shared
      sort and reads each tier's keys out of the column the merged segment
      will carry. Verified by the *same* `VerifySortedSegment` the flushed
      case uses, over a segment merged from eight overlapping sorted flushes
      with deletions.
- [ ] **`.fnm` claiming doc values the segment does not carry** (finding 14).
      Half-fixed: `field_updates` now accepts a doc-values-only field
      (finding 15), which was the blocker. What remains is zeroing
      `doc_values_type` in `fields_with_per_field_attributes` for fields no
      column was written for, plus a sweep of the three fixtures that could
      depend on the claim.
- [ ] **Richer `IndexSortField`** (b11 finding 2, unchanged): no-missing-value
      numeric sorts, arbitrary missing sentinels, and non-`MIN` multi-value
      selectors are still rejected rather than represented. Widening it is now
      cheaper than b11 estimated, because `sort_key_rank` is the single place
      the semantics live.
- [ ] **Parent field** (finding 8): with no parent-field write path, document
      blocks and an index sort cannot coexist. Java's rule needs
      `FieldInfo.parent_field` set on a NUMERIC doc-values field and the
      comparator wrapped in `parents.nextSetBit`.
- [ ] **One `Vec<PendingDoc>` instead of three parallel buffers** (Tier-2
      advisory): `pending_docs`, `pending_custom_freq_terms` and
      `pending_vectors` are hand-synchronised at eleven sites. Collapsing them
      into one struct makes the 1:1 invariant structural and reduces
      `sort_pending_buffer` to a single `permute_in_place`.
- [ ] **The index sort is not reachable from the FFI boundary** (Tier-2
      advisory): `crates/lucene-ffi/src/writer.rs` classifies every new error
      but exposes no `ffi_writer_set_index_sort` / `ffi_writer_add_doc_values_field`,
      so from OpenSearch the feature is still unreachable -- the same shape of
      problem this batch was opened to fix, one layer up. Precedent exists
      (`set_vector_field` is likewise Rust-only after c10), so this is a scope
      call, but it belongs on someone's list. Owner: whoever holds
      `lucene-ffi`.
- [ ] **Points** honour the sort trivially today because there is no
      points *flush* path at all (`IndexWriter` never writes `.kdd`/`.kdi`/
      `.kdm`); when one is added it must be built from the permuted buffer
      like every other format, and Java's
      `PointValuesWriter.MutableSortingPointValues` is then not needed.
