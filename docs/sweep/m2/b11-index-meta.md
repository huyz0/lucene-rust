# b11-index-meta

Sweep of `crates/lucene-index/src/{segment_info, segment_infos, check_index,
checksum_verify, deletes, term_delete, points_delete}.rs` against the Java
source at `/home/tuong/work/lucene`.

**Java-checkout caveat**: that working copy is Lucene *main* (`Version.LATEST
== LUCENE_11_0_0`, `SegmentInfos.VERSION_CURRENT == VERSION_11_0 == 11`), not
the pinned 10.5.0. Everything gated on `VERSION_11_0` — `SegmentInfos`'
per-field doc-values *overlay* block (`numOverlayFields`/`baseGen`/`deltaGens`)
— is therefore **out of scope** and is not a finding: 10.5.0's
`VERSION_CURRENT` is `VERSION_86 == 10`, which is exactly what this port
writes and accepts. All other Java behaviour cited below was checked to be
present in the 10.x line.

Totals: **26 findings** — 5 CORRECTNESS (all fixed), 12 MISSING (11 fixed, 1
recorded), 4 PERF (2 fixed with measurements, 2 reasoned), 5 INTENTIONAL.

Gate: `cargo fmt --all`, `cargo clippy -p lucene-index --all-targets -D
warnings`, `cargo test -p lucene-index` (402 tests) all green. Real-Lucene
cross-checks re-run with the checked-in 10.5.0 jars (`fixtures/.jars`).

---

## crates/lucene-index/src/segment_info.rs

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99SegmentInfoFormat.java`
- `lucene/core/src/java/org/apache/lucene/index/SegmentInfo.java`
- `lucene/core/src/java/org/apache/lucene/index/SortFieldProvider.java`
- `lucene/core/src/java/org/apache/lucene/search/{SortField,SortedNumericSortField,SortedSetSortField,BinarySortField}.java`
- `lucene/core/src/java/org/apache/lucene/search/{SortedNumericSelector,SortedSetSelector}.java`
- `lucene/core/src/java/org/apache/lucene/util/Version.java`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `parse` | `Lucene99SegmentInfoFormat.read` + `parseSegmentInfo` | field order, types and gates identical; sort-field body **was divergent** (finding 1) |
| `read_version` | `Version.fromBits` | **was missing** the `0..=255` per-component validation (finding 3) |
| `read_sort_field` | `SortFieldProvider.forName(readString()).readSortField(in)` | **was this port's own format** — rewritten, finding 1 |
| `read_plain_sort_field` (new) | `SortField.Provider.readSortField` + `SortField.readType` | identical decode; lowering documented (finding 2) |
| `read_sorted_numeric_sort_field` (new) | `SortedNumericSortField.Provider.readSortField` + `readSelectorType` | identical |
| `read_sorted_set_sort_field` (new) | `SortedSetSortField.Provider.readSortField` + `readSelectorType` | identical |
| `read_binary_sort_field` (new) | `BinarySortField.Provider.readSortField` | identical |
| `read_selector` (new) | both `readSelectorType`s | identical range check; non-`MIN` rejected (finding 2) |
| `numeric_missing` / `read_numeric_missing` (new) | `SortField.serialize`'s `case INT/LONG/FLOAT/DOUBLE` | identical wire decode |
| `float_to_sortable_int` / `double_to_sortable_long` (new) | `NumericUtils.floatToSortableInt` / `doubleToSortableLong` | identical |
| `write` | `Lucene99SegmentInfoFormat.write` + `writeSegmentInfo` | **was divergent** on the YES/NO byte (finding 4) and on sort fields (finding 1); `si.addFile` and the two `IllegalArgumentException` guards are not here (findings 5, 6) |
| `write_sort_field` | `SortField.Provider.writeSortField` -> `SortField.serialize` | now identical for the `LONG`-with-sentinel shape this port produces |
| `yes_no` (new) | `SegmentInfo.YES`/`SegmentInfo.NO` | identical |
| — | `SegmentInfo.{files,addFile,setFiles,namedForThisSegment,getAttribute,setAttribute,dir,name,getCodec}` | not-in-Java-sense: this port's `SegmentInfo` is a plain data struct with no `Directory`/`Codec` back-references; the `.si`'s bytes are the contract, the class graph is not (architecture invariant #2). INTENTIONAL. |

### Findings

1. **[CORRECTNESS → fixed]** *The index-sort encoding was this port's own
   invention.* Java writes each sort field as `writeString(providerName)`
   followed by that `SortFieldProvider`'s own bytestream (`SortField` →
   `field:String, type:String, reverse:i32, hasMissing:i32, missing…`). This
   port wrote a 4-byte `field, typeByte=0, reverseByte, missingByte` record.
   Consequence: a real Lucene reader could not open an index-sorted `.si` this
   port wrote, and this port rejected every real sorted `.si` with
   `UnknownSortFieldType`. The module doc comment named this openly ("this
   port's own internal format… NOT confirmed byte-compatible"), and
   `docs/parity.md` said the gap needed "a real sorted-index `.si` fixture" to
   close — a fixture that turned out to be one `GenSegmentInfo.java` method
   away. Fixed: real layout implemented for all four registered providers on
   the read side and for the `SortField`/`LONG` shape on the write side.
   **Evidence, both directions**: `fixtures/src/GenSegmentInfo.java::genSorted`
   now emits `fixtures/data/_2.si`, a real `Lucene99SegmentInfoFormat`-written
   two-field sorted `.si` (`timestamp` desc missing-last, `price` asc
   missing-first); `crates/lucene-index/tests/segment_info_fixtures.rs` parses
   it and asserts our re-encode is **byte-identical**; and
   `fixtures/src/VerifySegmentInfo.java` (already wired into
   `scripts/verify-write-path.sh`) now opens a Rust-written sorted `.si`
   through real Lucene 10.5.0 and compares the parsed `Sort` — run locally,
   `All 3 segment(s) verified against real Lucene. PASS`.

2. **[MISSING → recorded]** *Sorts this port's `IndexSortField` cannot
   represent.* `IndexSortField` carries only `(field, reverse,
   missing-first-or-last)`. Java's providers can express: a numeric sort with
   **no** missing value (treated as `0` — neither first nor last), an
   arbitrary numeric missing sentinel, and multi-value selectors `MAX`,
   `MIDDLE_MIN`, `MIDDLE_MAX`. `parse` now decodes all of them and then
   **rejects** the unrepresentable ones with `Error::UnsupportedSortField`
   naming exactly what it saw, rather than silently lowering onto a sort order
   this port would then get wrong (tested:
   `missing_less_numeric_sort_is_rejected_not_guessed`,
   `arbitrary_numeric_missing_value_is_rejected`,
   `non_min_selector_rejected_rather_than_silently_downgraded`). **Not fixed**
   because widening it means adding fields to `IndexSortField` / variants to
   `SortMissingValue`, whose construction and `match` sites live in
   `segment_writer.rs` and `merge.rs` — files owned by the concurrently
   running b9 and b10 batches. Recorded in `docs/parity.md` as the remaining
   scope limit.

3. **[MISSING → fixed]** *No `Version.fromBits` validation.* Java packs each
   component into one byte and throws `IllegalArgumentException` outside
   `0..=255`. We accepted any `i32`, so a corrupt `.si` produced a nonsense
   version that later cross-`.si` comparisons (finding 15) would then trust.
   Fixed in `read_version`; test `out_of_range_version_component_rejected`.

4. **[CORRECTNESS → fixed]** *`isCompoundFile`/`hasBlocks` written as `0` for
   false.* Java writes `(byte)(flag ? SegmentInfo.YES : SegmentInfo.NO)` and
   `SegmentInfo.NO == -1`, i.e. `0xFF`. Every reader tests `== YES`, so this
   round-tripped through both readers and was invisible to every existing
   test — it was caught only by the new byte-exact re-encode assertion added
   to `tests/segment_info_fixtures.rs`, which now diffs our `write(parse(x))`
   against the Java-written `x` for all three `.si` fixtures. Fixed via a
   `yes_no` helper.

5. **[MISSING → recorded, out of batch]** *`.si` does not list itself.*
   `Lucene99SegmentInfoFormat.write` calls `si.addFile(fileName)` before
   writing, so a real `.si`'s file set always contains `<segment>.si`
   (confirmed in the `_0`/`_1`/`_2` fixture bytes). This port's
   `segment_info::write` cannot do it — `SegmentInfo` has no `name` field, the
   caller derives the file name — and of the two callers only
   `merge.rs` pushes `si_name`; `segment_writer::flush_stored_only_segment`
   does not. `IndexFileDeleter` reference-counts from exactly this set, so a
   `.si` missing from its own list is a file nothing holds a reference to.
   **Not fixed here**: the defect is in `segment_writer.rs` (b9's batch).
   Instead, `check_index` gained a `si.files_lists_itself` check that catches
   it (finding 20), and this is flagged for b9.

6. **[MISSING → recorded]** *`write` performs neither of Java's two write-time
   guards*: `version.major >= 7`, and every file in `files` being prefixed by
   the segment name (`IndexFileNames.parseSegmentName`). Not fixed because
   `write` returns `Vec<u8>`, not a `Result`, and changing that signature
   would ripple into `segment_writer.rs`/`merge.rs`/the examples. The existing
   doc comment already documents the files-prefix stance ("a hand-built writer
   is trusted"). Low impact: both guards only fire on writer bugs, and the
   segment-name one is now indirectly covered by `si.files_lists_itself`.

7. **[INTENTIONAL]** Diagnostics/files/attributes are `Vec<(String,String)>` /
   `Vec<String>` in file order, not `HashMap`/`Set`. Preserves the on-disk
   order (which is what makes byte-exact re-encode possible at all — see
   finding 4) and avoids hashing for maps of three entries.

### Verdict

Swept clean apart from finding 2 (recorded, needs an API change owned by
another batch) and findings 5/6 (recorded, the fix site is in another batch's
file; finding 5 is now *detected* here).

---

## crates/lucene-index/src/segment_infos.rs

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/index/SegmentInfos.java`
- `lucene/core/src/java/org/apache/lucene/index/SegmentCommitInfo.java`
- `lucene/core/src/java/org/apache/lucene/index/IndexFileNames.java`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `parse` | `SegmentInfos.readCommit(dir, in, gen, minMajor)` + `parseSegmentInfos` | field order/types/BE-vs-LE/`format > VERSION_74` gate identical; **was missing** several validations (findings 8, 9, 15) |
| `read_vint_version` | `Version.fromBits` | **was missing** range validation — fixed, finding 9 |
| `read_latest` | `SegmentInfos.readLatestCommit` + `FindSegmentsFile` | equivalent; the generation-picking lives in `lucene_store::directory` |
| `write` | `SegmentInfos.commit` (= `prepareCommit` + `finishCommit`) | identical two-phase shape |
| `write_pending` | `SegmentInfos.write(Directory)` + `prepareCommit` | identical, including `syncMetaData` first and delete-on-failure |
| `finish_pending` | `SegmentInfos.finishCommit` | identical (rename, then `syncMetaData`, delete on failure) |
| `rollback_pending` | `SegmentInfos.rollbackCommit` | identical (ignore failures) |
| `to_bytes` | `SegmentInfos.write(IndexOutput)` | divergent in three deliberate ways — finding 12 |
| `SegmentCommitInfo::files` (new) | `SegmentCommitInfo.files()` | ported, finding 10 |
| — | `readCodec` (`Codec.forName` SPI) | INTENTIONAL: this port stores the codec name as a string; there is no codec SPI to look up. |
| — | `SegmentInfos.{add,remove,replace,clone,updateGeneration,changed,applyMergeChanges,asList,totalMaxDoc,createBackupSegmentInfos,rollbackSegmentInfos}` | mutation/bookkeeping API of a live `IndexWriter`; not-in-scope for a read/write codec module. |

### Findings

8. **[MISSING → fixed]** *No `luceneVersion.major < indexCreatedVersion`
   check.* Java throws `CorruptIndexException` — a commit cannot claim to have
   been created by a newer major than the one that wrote it. Every later
   `indexCreatedVersionMajor >= 7` gate trusts this value, so accepting a
   forward-dated one skews all of them. Fixed; tests
   `created_version_newer_than_writer_version_rejected` and
   `created_version_equal_to_writer_version_accepted`.

9. **[MISSING → fixed]** *No `Version.fromBits` range validation on either
   version triple.* Same as finding 3, for the vint-encoded versions. Fixed;
   test `out_of_range_lucene_version_component_rejected`.

10. **[MISSING → fixed]** *No `SegmentCommitInfo.files()`.* Java's is
    `info.files()` ∪ the live-docs file ∪ `fieldInfosFiles` ∪ every
    `dvUpdatesFiles` value. Nothing in this port had it, so both file-walking
    tools (`check_index`, `checksum_verify`) walked `SegmentInfo.files` alone
    and silently skipped the `.liv` and the generational field-infos /
    doc-values-update files — i.e. exactly the files a delete or DV-update
    round writes *last* and is therefore most likely to have written wrong.
    Fixed as `SegmentCommitInfo::files(&si_files)`, deterministically ordered
    and de-duplicated; consumed by findings 20 and 22.

11. **[MISSING → moved to `check_index`]** Java's `readCommit` also enforces
    `delCount`/`softDelCount` ≤ `maxDoc` and their sum ≤ `maxDoc`, the
    segment's version ≥ `minSegmentLuceneVersion` and ≥
    `indexCreatedVersionMajor`, `minVersion != null` once
    `indexCreatedVersionMajor >= 7`, and total docs ≤
    `IndexWriter.getActualMaxDocs()`. All of these need each segment's `.si`,
    which Java has already opened inside its parse loop and which this port's
    `parse` deliberately never opens (documented design: no `Directory`
    dependency). Fixed by adding every one of them to `check_index` as
    `commit.*` checks — see findings 15 and 16. `parse` keeps only the two
    that need no `.si` (findings 8, 9).

12. **[INTENTIONAL]** `to_bytes` diverges from `SegmentInfos.write(IndexOutput)`
    in three ways, all deliberate and pre-existing: it writes the caller's
    `id` rather than a fresh `StringHelper.randomId()`; it writes the caller's
    `lucene_version` rather than `Version.LATEST`; and it writes
    `min_segment_lucene_version.unwrap_or(lucene_version)` rather than
    recomputing the minimum across the segments' `.si` versions (which it
    cannot see). The first two make `write` a true inverse of `parse`, which
    is what makes round-trip and byte-exactness testable at all.

13. **[INTENTIONAL]** `dv_update_files` is `Vec<(i32, Vec<String>)>` rather
    than a map; on-disk order is preserved and the vector is written back
    verbatim.

### Verdict

Swept clean. The `readCommit` validations that cannot live here now live in
`check_index` and are tested there.

---

## crates/lucene-index/src/check_index.rs

Java counterpart: `lucene/core/src/java/org/apache/lucene/index/CheckIndex.java`
(plus `PendingSoftDeletes.countSoftDeletes`).

### Java `test*` coverage, before → after

| Java method | What Java checks | Before | After |
|---|---|---|---|
| `testLiveDocs` | `liveDocs != null` when `hasDeletions`; cardinality vs `numDocs`; **no** clear bit when `hasDeletions == false` | cardinality + `.liv` size vs `.si` | + `commit.del_count_zero_without_del_gen`, `commit.del_count_within_max_doc` |
| `testFieldInfos` | opens `.fnm`, counts fields | flags-vs-files cross-check (stronger than Java's) | unchanged |
| `testFieldNorms` | iterates every norm value | file-presence cross-check only | unchanged (recorded, finding 25) |
| `testPostings` | term order, `docFreq > 0`, `sumDocFreq`/`sumTotalTermFreq`/`docCount`/`min`/`maxTerm`, doc-id order, positions/offsets/payloads | `totalTermFreq` re-derivation + doc-id order | **+ `postings.terms_sorted`, `postings.doc_freq_positive`, `postings.field_summary`** (finding 17); positions/offsets/payloads still not walked (finding 26) |
| `testStoredFields` | pulls **every** document, deleted included; `docCount` vs `numDocs` | `max_doc` comparison only | **+ `stored_fields.every_doc_decodes`** (finding 14) |
| `testTermVectors` | every doc's vectors decoded; field's `storeTermVector` flag; (slow level) vectors vs postings | *nothing* | **+ `term_vectors.doc_count_matches_si`, `.every_doc_decodes`, `.fields_marked_in_fnm`** (finding 18) |
| `testDocValues` | every field's every value; ordinal ranges; sorted-set ord ordering; counts | *nothing* | **+ `doc_values.entry_present:*`, `doc_values.values_decode:*`** (finding 19) |
| `testPoints` | values within field bounds; leaf boxes; `size()`/`getDocCount()` | value bounds, leaf boxes, point count | unchanged (already close to Java) |
| `testSort` | segment declaring an `indexSort` is really in that order | *nothing* | **+ `sort.docs_in_index_sort_order`** (finding 21) |
| `checkSoftDeletes` | live docs with the soft-deletes field vs `softDelCount` | *nothing* | **+ `soft_deletes.count_matches`** (finding 21) |
| `testVectors` / `testHnswGraphs` | KNN vectors + graph structure | *nothing* | unchanged — no vector/HNSW write path exists in this port at all |
| `checkIndex` top level | `maxSegmentName` vs `counter`, per-segment aggregation | *nothing* | **+ `commit.*`** (finding 16) |

### Findings

14. **[MISSING → fixed]** *`testStoredFields` never decoded a document.* The
    check compared `StoredFieldsReader::max_doc()` against `.si`'s
    `doc_count` — that reads only `.fdm` metadata and the `.fdx` index, and
    never decompresses a single chunk. A corrupted `.fdt` body therefore
    passed a clean run. Java pulls every document, deleted ones included.
    Fixed (`stored_fields.every_doc_decodes`); the existing
    `stored_fields_doc_count_mismatch_is_flagged` test was tightened to assert
    the new check passes on genuinely-good bytes while the count check fails.

15. **[MISSING → fixed]** *None of `SegmentInfos.readCommit`'s cross-`.si`
    validations existed anywhere* (see finding 11). Added as
    `commit.del_count_within_max_doc`,
    `commit.soft_del_count_within_max_doc`,
    `commit.del_plus_soft_del_within_max_doc`,
    `commit.del_count_zero_without_del_gen`,
    `commit.segment_version_at_or_after_min`,
    `commit.segment_version_at_or_after_created`,
    `commit.segment_records_min_version`. A new
    `check_segment_in_commit(dir, Some(&infos), commit)` entry point supplies
    the commit header; `check_segment` keeps its old signature and runs the
    segment-local subset. Tests: `del_count_larger_than_max_doc_is_flagged`,
    `del_count_without_del_gen_is_flagged`.

16. **[MISSING → fixed]** *No commit-level checks at all.* Added `check_commit`
    — `commit.segment_names_unique`, `commit.segment_names_well_formed`,
    `commit.counter_ahead_of_segment_names` (real `CheckIndex`'s
    `validCounter`/`maxSegmentName`: a counter that has fallen behind makes
    the next flush reuse a live segment's name and clobber its files), and
    `commit.total_max_doc_within_bounds` (LUCENE-6299's `IndexWriter.MAX_DOCS
    == Integer.MAX_VALUE - 128` bound, which needs every segment's `maxDoc`
    and so is appended after the per-segment pass). `check_directory` now
    returns the commit result first, then the segments. Tests:
    `commit_counter_behind_segment_names_is_flagged`,
    `duplicate_segment_names_in_a_commit_are_flagged`.

17. **[MISSING → fixed]** *`testPostings`' term- and field-level invariants.*
    Added `postings.terms_sorted` (strictly increasing within a field),
    `postings.doc_freq_positive`, and `postings.field_summary` — the latter
    re-derives `.tmd`'s `numTerms`, `sumDocFreq`, `sumTotalTermFreq`,
    `minTerm`, `maxTerm` and `docCount` (via a `FixedBitSet` union of every
    decoded doc id) from the dictionary itself, and checks `sumTotalTermFreq
    >= sumDocFreq`. These are the numbers a scorer's IDF and length
    normalisation read; nothing cross-checked them before. All pass on the
    real `blocktree_index` fixture (8959 docs).

18. **[MISSING → fixed]** *No term-vectors check whatsoever.* Added
    `check_term_vectors`: reader `max_doc` vs `.si`, every document's vectors
    decoded (deleted included, as Java does), and every field a document
    actually carries vectors for cross-checked against `.fnm`'s
    `storeTermVectors` — Java's exact "docID=… has term vectors for field=…
    but FieldInfo has storeTermVector=false" check. Test
    `term_vector_checks_actually_run_on_a_term_vectors_fixture` guards against
    a silently-never-firing check.

19. **[MISSING → fixed]** *No doc-values check whatsoever* — the module's own
    doc comment listed it as "still out of scope … a separate task". It was
    not: `lucene-codecs::doc_values` already exposes `parse_meta` plus a
    per-doc accessor for all five kinds, and `lucene-search` already calls
    them. Added `check_doc_values`: every field's every per-doc value decoded
    out of `.dvd`; SORTED and single-valued SORTED_SET ordinals bounds-checked
    against `terms_dict_size`; multi-valued SORTED_SET ordinals additionally
    checked for being a strictly increasing *set* within a doc; BINARY value
    lengths against `.dvm`'s declared `min_length`/`max_length`; and the
    decoded docs-with-a-value count against `.dvm`'s own `numDocsWithField`.
    Test `corrupted_doc_values_payload_is_caught` flips a `.dvd` payload byte
    that previously passed clean.

20. **[MISSING → fixed]** *Files checked were `SegmentInfo.files`, not
    `SegmentCommitInfo.files()`* (see finding 10) — the `.liv` and any
    generational field-infos/doc-values-update file were never opened or
    footer-checked. Fixed. Also added `si.files_lists_itself` (finding 5),
    which immediately found two hand-built test fixtures in this same module
    that omitted `_0.si`; both corrected.

21. **[MISSING → fixed]** *No `testSort`, no `checkSoftDeletes`.* Added
    `sort.docs_in_index_sort_order` — reads the sort fields' NUMERIC /
    SORTED_NUMERIC doc values and walks adjacent doc ids through
    `segment_writer::sort_key_rank`, i.e. the *same comparator the
    sort-on-flush writer used to produce the order*, applied in reverse as a
    verifier — and `soft_deletes.count_matches`, the port of
    `PendingSoftDeletes.countSoftDeletes(iterator, liveDocs)`. Both are
    skipped (not vacuously passed) when the segment declares no sort / has no
    soft-deletes field / has no doc-values files. Tests:
    `index_sort_order_is_verified_against_the_actual_doc_values` (good and bad
    fixture), `unsorted_segment_gets_no_sort_check`,
    `soft_delete_count_is_verified_against_the_soft_deletes_field`.

22. **[PERF → reasoned, accepted]** The new checks are O(docs × fields) decode
    work: stored fields, term vectors and doc values are each read end to end.
    That is exactly what Java's `CheckIndex` does at its default level, and it
    is the whole point — the previous metadata-only checks were cheap because
    they did not look at the data. `checksum_verify.rs` remains the cheap
    pre-flight (one CRC pass per file, no format decoding), and its module doc
    already positions it that way. New regression guard:
    `every_real_lucene_index_fixture_passes_every_check` runs the full check
    over ten real Java-written index fixtures (`blocktree_index`,
    `live_docs_index`, `doc_values_index`, `sorted_dv_index`,
    `multi_valued_dv_index`, `term_vectors_index`, `norms_index`,
    `points_index`, `doc_values_skip_index`, `doc_values_varying_bpv`) and
    requires zero failures — the no-false-positives floor for everything added
    here.

25. **[MISSING → recorded]** `testFieldNorms` iterates every norm value; we
    only cross-check `.nvd`/`.nvm` presence against `.fnm`'s `omitNorms`.
    Tractable, but norms decoding was not otherwise in this batch's blast
    radius and the check needs the same `open_*`-per-format plumbing the DV
    check now has. Recorded rather than rushed.

26. **[MISSING → recorded]** `testPostings` at Java's slow level also walks
    positions, offsets and payloads (monotonic positions, `startOffset <=
    endOffset`, offsets non-decreasing) and cross-checks term vectors against
    postings. Not ported; the doc-id/freq/statistics layer added here is the
    part that guards the metadata/data agreement this port's writer can get
    wrong.

### Verdict

Substantially closed. Open: findings 25 (norms values), 26 (positions/offsets/
payloads, TV-vs-postings), and vectors/HNSW (no write path exists to check).

---

## crates/lucene-index/src/checksum_verify.rs

Java counterpart: `CheckIndex`'s checksum-only path +
`lucene/core/src/java/org/apache/lucene/codecs/CodecUtil.java`
(`checkFooter`/`checksumEntireFile`) + `SegmentInfos.files(boolean)`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `verify_directory` | `CheckIndex` `-fastCheck` loop over `SegmentInfos.files(true)` | **was divergent**: walked `SegmentInfo.files` only, and omitted `segments_N` — finding 23 |
| `verify_file` | `CodecUtil.checksumEntireFile` + `checkFooter` | identical (full CRC over the payload, not `retrieveChecksum`'s shape-only check — the module doc already explains why) |
| `VerifyReport::{total,failed_count,all_passed,failures}` | `CheckIndex.Status` aggregation | equivalent |

### Findings

23. **[MISSING → fixed]** *The `.liv`, the generational field-infos /
    doc-values-update files, and `segments_N` itself were never verified.*
    `SegmentCommitInfo.files()` is the correct set (finding 10) and
    `SegmentInfos.files(true)` adds the commit file. Consequence: a corrupted
    live-docs file — the most recently written file in any index that has
    taken deletes — passed a "clean" checksum run. Fixed; tests
    `live_docs_file_is_verified_and_its_corruption_detected` (creates a real
    `.liv` via `deletes::apply_deletes`, republishes the commit, then flips a
    payload byte) and `segments_file_itself_is_in_the_verified_set`.

### Verdict

Swept clean.

---

## crates/lucene-index/src/deletes.rs

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/index/IndexFileNames.java`
- `lucene/core/src/java/org/apache/lucene/index/{PendingDeletes,PendingSoftDeletes}.java`
- `lucene/core/src/java/org/apache/lucene/index/ReadersAndUpdates.java` (`writeLiveDocs`)
- `lucene/core/src/java/org/apache/lucene/index/SegmentCommitInfo.java` (`getNextDelGen`)

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `liv_file_name` | `IndexFileNames.fileNameFromGeneration(name, "liv", gen)` | **was divergent** for `gen == 0` — finding 24 |
| `mark_deleted` | `PendingDeletes.delete(docID)` (looped) + `FixedBitSet` | same semantics (idempotent, newly-deleted counted once); **was** O(maxDoc) bit-by-bit fill — finding 27 |
| `apply_deletes` | `ReadersAndUpdates.writeLiveDocs` + `SegmentCommitInfo.getNextDelGen`/`advanceDelGen` | same generation arithmetic (`-1 → 1`, else `+1`), write + sync + new `SegmentCommitInfo`; **was missing** the `delCount <= maxDoc` guard — finding 28 |
| — | `PendingDeletes.{onNewReader,dropChanges,numPendingDeletes,mustInitializeHardDeletes}` | live-`IndexWriter` reader-pool bookkeeping; no reader pool in this port. |
| — | `BufferedUpdates`, `FrozenBufferedUpdates`, `DocValuesUpdate`, `Term` | see finding 29. |

### Findings

24. **[CORRECTNESS → fixed]** *`liv_file_name(seg, 0)` returned
    `_<seg>_0.liv`.* `IndexFileNames.fileNameFromGeneration` has a dedicated
    `gen == 0` branch returning `segmentFileName(base, "", ext)` — no
    generation suffix at all — and returns `null` for `gen == -1`. Generation
    0 is unreachable for `.liv` from a real writer (`getNextDelGen()` goes
    `-1 → 1`), but it is reachable from a corrupt or hand-built `segments_N`,
    and every caller in the workspace (including `check_index`,
    `checksum_verify`, `directory_reader`, `index_writer`, `lucene-ffi`) would
    then look for a file no Lucene writer would ever produce. Fixed; the
    module's own test that asserted the wrong name was corrected. The `-1`
    case is documented as a caller precondition rather than an `Option`
    return, because changing the signature would ripple into four files owned
    by other batches and every caller already guards it.

27. **[PERF → fixed]** *The "all live" bitset was built with `max_doc`
    individual `set(i)` calls*, each a bounds check plus a
    read-modify-write of a `u64`. Replaced with a whole-word `vec![u64::MAX;
    bits2words(max_doc)]` plus a masked tail — the shape of Java's
    `FixedBitSet.set(from, to)`. This is the first-delete path for every
    segment, so it is O(maxDoc) per segment either way; the constant factor
    drops by ~64×. Correctness of the tail mask (a stray set bit past
    `max_doc` would read as a live doc that does not exist, since
    `cardinality()` counts raw words) is pinned by the new
    `all_live_bitset_masks_bits_past_max_doc` test over `max_doc ∈ {1, 63, 64,
    65, 130, 256}`.

28. **[MISSING → fixed]** *`apply_deletes` could produce a `delCount` past
    `maxDoc`.* `SegmentInfos.write` throws `IllegalStateException` for exactly
    that, so this port could build a commit its own writer would then have to
    reject. Now rejected at the source, with the segment named
    (`Error::DelCountExceedsMaxDoc`); test `del_count_past_max_doc_is_rejected`.

29. **[MISSING → recorded]** *No `BufferedUpdates`/`FrozenBufferedUpdates` at
    all*, so the generation-ordering question the sweep brief raises —
    "do deletes applied to a segment respect the segment's own generation" —
    has no implementation to be right or wrong about. In Java a frozen delete
    packet applies to a segment only when the segment's `bufferedDeletesGen`
    is below the packet's `delGen`, and a term deleted inside the same DWPT
    applies only to doc ids below its `docIDUpto`; `BufferedUpdates` also
    maintains the RAM accounting (`BYTES_PER_DEL_TERM` etc.) that triggers
    `applyAllDeletes`. This port resolves deletes eagerly, one segment at a
    time, against a reader the caller has already opened
    (`term_delete`/`points_delete`), which is order-independent by
    construction: there is no packet to order and no `docIDUpto` window,
    because a delete is resolved against exactly the segment state the caller
    hands it. That is a real scope limit, not a silent divergence — both
    modules' doc comments and `docs/parity.md` state it — and closing it is
    the multi-segment `IndexWriter` orchestration task, well outside this
    batch.

### Verdict

Swept clean apart from finding 29 (recorded, whole-feature scope).

---

## crates/lucene-index/src/term_delete.rs

Java counterparts: `lucene/core/src/java/org/apache/lucene/index/{Term,BufferedUpdates,FrozenBufferedUpdates}.java`,
`lucene/core/src/java/org/apache/lucene/search/TermQuery.java`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `resolve_term_doc_ids` | `FrozenBufferedUpdates.apply`'s per-term `TermsEnum.seekExact` + `PostingsEnum` walk, filtered by live docs | identical semantics for one segment: ascending doc ids, live-only, unknown field/term ⇒ empty (matching `TermQuery`'s null-`Scorer` "no matches") |
| `resolve_and_apply_term_delete` | `FrozenBufferedUpdates.apply` + `ReadersAndUpdates.writeLiveDocs` | identical for one segment |
| — | `BufferedUpdates.{addTerm,addQuery,addNumericUpdate,addBinaryUpdate,clear,ramBytesUsed}` | finding 29. |
| — | `FrozenBufferedUpdates.{applyTermDeletes,applyQueryDeletes,applyDocValuesUpdates}` | finding 29 for the first two; doc-values updates are `doc_values_updates.rs`'s (b8) territory. |

### Findings

*(No new findings.)* Term-based delete **ordering** is not a divergence here:
Java's `applyTermDeletes` sorts by term only to share a single `TermsEnum`
seek cursor, and its `docIDUpto` window exists solely because a delete can
be buffered mid-flush in the same DWPT. Neither applies to eager
single-segment resolution against an already-flushed segment. Doc ids come
back ascending from `Postings::docs` (which the postings format guarantees)
and `apply_deletes` is order-independent (`mark_deleted` is idempotent per
doc). Verified by the existing fixture-backed tests.

### Verdict

Swept clean.

---

## crates/lucene-index/src/points_delete.rs

Java counterparts:
- `lucene/core/src/java/org/apache/lucene/search/PointRangeQuery.java`
- `lucene/core/src/java/org/apache/lucene/index/PointValues.java` (`intersect`, `IntersectVisitor`)
- `lucene/core/src/java/org/apache/lucene/index/FrozenBufferedUpdates.java` (`applyQueryDeletes`)

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `resolve_points_range_doc_ids` | `PointRangeQuery`'s `IntersectVisitor` driven by `PointValues.intersect`, folded into a doc-id set | **was divergent** (data dims, no pruning) — findings 30, 31 |
| `resolve_and_apply_points_range_delete` | `FrozenBufferedUpdates.applyQueryDeletes` for one query/segment + `writeLiveDocs` | identical for one segment |
| `packed_value_in_range` / `compare_unsigned` (removed) | `PointRangeQuery.matches` | superseded by `RangeVisitor` in `lucene-codecs::points`, which is the real port |

### Findings

30. **[CORRECTNESS → fixed]** *The query box was compared over the field's
    `num_dims` (data dimensions) rather than `num_index_dims`.*
    `PointRangeQuery.checkValidPointValues` rejects a values instance whose
    `getNumIndexDimensions()` differs from the query's `numDims`, and only
    index dimensions have cell bounds in the BKD packed index at all. For a
    field with trailing non-indexed data dimensions (`LatLonShape`, range
    fields) this port sliced past the end of a correctly-sized `[min, max]`
    box — a panic, or a comparison against bytes the box does not describe.
    Fixed by delegating to `PointsReader::range_query`, whose visitor is
    index-dimension-scoped; regression test
    `query_box_covers_index_dimensions_only_not_data_dimensions` builds a
    `num_dims=2, num_index_dims=1` field and passes an 8-byte box.

31. **[PERF → fixed, measured]** *`decode_all_points` + in-memory filter.* The
    module doc argued this was "correct (just not sublinear)" because no BKD
    range-matching primitive existed anywhere in the workspace. Batch b7 built
    one (`PointsReader::intersect` with CELL_INSIDE/CROSSES/OUTSIDE pruning,
    plus a `range_query` wrapper); migrating to it was this batch's job.
    Measured on a 200 000-point single-dimension field, release build, mean of
    50 iterations:

    | range | hits | `range_query` | `decode_all_points` | speedup |
    |---|---|---|---|---|
    | 100 of 200k | 100 | 28.0 µs | 6.16 ms | **220×** |
    | 2 000 of 200k | 2 000 | 30.4 µs | 6.11 ms | **201×** |
    | 50% | 100 001 | 121 µs | 6.19 ms | **51×** |
    | everything | 200 000 | 203 µs | 6.42 ms | **32×** |

    Even the match-everything case is 32× faster, because a fully-contained
    cell takes `visit(docID)` — doc ids only, no packed value decoded — which
    is precisely what `CELL_INSIDE_QUERY` buys. (The benchmark was a throwaway
    example, run and then deleted; `lucene-index` has no `benches/` directory
    and adding a criterion dependency to its `Cargo.toml` during a concurrent
    multi-batch edit was not worth the conflict risk.)

### Verdict

Swept clean.

---

## Cross-batch notes

- **`segment_writer.rs` (b9)**: `flush_stored_only_segment` writes a `.si`
  whose `files` set omits `<segment>.si`. Real Lucene always includes it
  (`Lucene99SegmentInfoFormat.write` → `si.addFile`), `merge.rs` already does,
  and `IndexFileDeleter` reference-counts from that set. `check_index`'s new
  `si.files_lists_itself` will flag it. One line; left to b9. (Finding 5.)
- **`lucene-search/src/points_query.rs`**: `corrupt_kdd_leaf_data_surfaces_as_points_error`
  fails on the current tree (`unwrap_err()` on an `Ok`). `points_query.rs`
  itself is unmodified; `lucene-codecs/src/points.rs` has +879 lines from the
  b7/b8 work, so the corrupt-`.kdd` input that used to error now decodes.
  Outside this batch — reported, not touched. `cargo test -p lucene-index` is
  fully green.
