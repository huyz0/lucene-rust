# b9-index-write

Sweep of the `lucene-index` write path against Lucene 10.5.0 at
`/home/tuong/work/lucene`.

Files swept:

- `crates/lucene-index/src/index_writer.rs`
- `crates/lucene-index/src/segment_writer.rs`
- `crates/lucene-index/src/indexing_chain.rs`
- `crates/lucene-index/src/update_document.rs`
- `crates/lucene-index/src/lib.rs`

Java counterparts compared against (all under
`/home/tuong/work/lucene/lucene/core/src/java/org/apache/lucene/index/`):
`IndexWriter.java`, `IndexWriterConfig.java`, `LiveIndexWriterConfig.java`,
`DocumentsWriter.java`, `DocumentsWriterPerThread.java`,
`DocumentsWriterFlushControl.java`, `DocumentsWriterDeleteQueue.java`,
`DocumentsWriterPerThreadPool.java`, `FlushByRamOrCountsPolicy.java`,
`IndexingChain.java`, `TermsHash.java`, `TermsHashPerField.java`,
`FreqProxTermsWriter.java`, `FreqProxTermsWriterPerField.java`,
`StoredFieldsConsumer.java`, `SortingStoredFieldsConsumer.java`,
`NormValuesWriter.java`, `SegmentWriteState.java`, `FieldInfos.java`,
`IndexFileDeleter.java`, `SegmentCommitInfo.java`, `SegmentInfos.java`,
`ByteBlockPool`/`IntBlockPool`/`ByteSlicePool`, plus
`org/apache/lucene/store/{Directory,FSDirectory}.java` and
`org/apache/lucene/document/{Document,Field,FieldType}.java`.

Two files outside the batch were changed, because the fixes below are not
expressible without them and neither is another batch's in-flight work:
`crates/lucene-store/src/{directory.rs,index_output.rs}` (the
`rename`/`deleteFile`/`syncMetaData` primitives a two-phase commit needs, plus
`pending_segments_file_name`), `crates/lucene-index/src/segment_infos.rs`
(`write_pending`/`finish_pending`/`rollback_pending`), and one error-mapping
arm in `crates/lucene-ffi/src/writer.rs` for the two new error variants.

---

## `crates/lucene-index/src/lib.rs`

No Java counterpart — it is the crate root's `mod` list plus
`#![forbid(unsafe_code)]`. Every module named there exists. Nothing to sweep.

### Verdict

Swept clean.

---

## `crates/lucene-index/src/indexing_chain.rs`

Java: `IndexingChain.java` (`processDocument`, `processField`,
`PerField.invert`), `TermsHashPerField.java` (`add`, `writeProx`,
`writeOffsets`), `FreqProxTermsWriterPerField.java`,
`ByteBlockPool`/`IntBlockPool`/`ByteSlicePool`.

| Rust | Java | Status |
|---|---|---|
| `invert_documents` | `IndexingChain.processDocument` + `PerField.invert` + `TermsHashPerField.add` | divergent (same output shape, entirely different memory shape — see F-8/F-9) |
| `InMemoryInvertedIndex` / `TermKey` | `FreqProxFields` view over `TermsHash`'s pools | divergent (owned `BTreeMap<(String,String), Vec<PostingEntry>>` vs. `BytesRefHash` + block pools) |
| `PostingEntry::{term_freq, positions, offsets}` | `FreqProxPostingsArray.termFreqs` / `ByteSliceReader` over the prox stream | identical results |
| `InMemoryInvertedIndex::postings` | `Terms.iterator().seekExact` | not-in-Java (test/caller convenience) |
| — | `IndexingChain.processField` per-field `FieldInfo` reconciliation (`FieldInfos.Builder`, global field numbers) | not-in-Java by design: the field list is fixed at `IndexWriter::open`, so there is no global field-number map to reconcile (F-18) |
| — | `PerField.invert`'s position/offset validation | missing but unreachable (see F-14) |
| — | `Analyzer.getPositionIncrementGap`/`getOffsetGap` between values of a multi-valued field | missing but unreachable — this API carries one text per `(doc, field)` |
| — | `TermVectorsConsumerPerField`, `PointValuesWriter`, `DocValuesWriter`, `VectorValuesWriter` fan-out from the same invert pass | partially fixed by F-8; points/vectors have no write path at all |

---

## `crates/lucene-index/src/segment_writer.rs`

Java: `DocumentsWriterPerThread.flush`/`sealFlushedSegment`,
`StoredFieldsConsumer`, `SortingStoredFieldsConsumer`, `SegmentWriteState`,
`SegmentInfo`/`SegmentCommitInfo` construction in `DocumentsWriterPerThread`.

| Rust | Java | Status |
|---|---|---|
| `flush_stored_only_segment` | `DocumentsWriterPerThread.flush` + `sealFlushedSegment` | divergent by scope: stored fields + `.fnm` + `.si` only; `IndexWriter` layers postings/TV/DV/norms on top afterwards, re-patching the `.si` each time (F-16) |
| `flush_sorted_stored_only_segment` | `DocumentsWriterPerThread` sort-on-flush (`SortingStoredFieldsConsumer`) | divergent: sort keys are supplied in memory, not read back from a `DocValuesProducer`; documented in `docs/parity.md` |
| `sort_key_rank` | `SortField.getComparator` + `setMissingValue` | identical semantics |
| `write_file` | `Directory.createOutput` + `close` | identical |
| — | `SegmentCommitInfo` id (`StringHelper.randomId()`) | divergent, port-wide (F-13) |
| — | `abort()` (delete every file the aborted DWPT created) | missing, port-wide (F-11) |

---

## `crates/lucene-index/src/update_document.rs`

Java: `IndexWriter.updateDocument(Term, doc)` +
`DocumentsWriterDeleteQueue.add` + `BufferedUpdatesStream.applyDeletesAndUpdates`.

| Rust | Java | Status |
|---|---|---|
| `update_document` | `IndexWriter.updateDocument(Term, Iterable<IndexableField>)` | divergent: resolves and applies the delete eagerly against caller-supplied segments and commits immediately, where Java buffers it in the delete queue and applies it at the next flush. Atomicity of the *result* matches (one `segments_N` write at the end) |
| `SegmentDeleteSource` | `ReaderPool.get(info)` | not-in-Java (there is no reader pool here) |
| — | delete-queue sequence numbers | missing (F-7) |
| — | `softUpdateDocument` | missing (F-12) |

---

## `crates/lucene-index/src/index_writer.rs`

Java: `IndexWriter.java` primarily, plus `IndexWriterConfig`/
`LiveIndexWriterConfig` for the configuration surface.

### Method correspondence

| Rust | Java | Status |
|---|---|---|
| `open` | `IndexWriter(Directory, IndexWriterConfig)` | divergent (no `IndexWriterConfig`, no write lock, no `IndexFileDeleter`, no `IndexDeletionPolicy`) |
| `add_document` | `addDocument` | divergent: returns `()` not a sequence number; never auto-flushes (F-7, F-10) |
| `add_document_with_custom_freq_terms` | — | not-in-Java (this port's `DocsAndCustomFreqs` entry point) |
| `update_document` | `updateDocument(Term, doc)` | divergent + two bugs fixed (F-2, F-4) |
| `delete_documents` | `deleteDocuments(Term...)` | divergent: eager + immediately committed; one bug fixed (F-4) |
| `delete_all` | `deleteAll()` | **newly ported** (F-6) |
| `commit` | `commit()` | divergent: always writes a new generation even with nothing changed (no `changeCount`); otherwise `prepareCommit` + `finishCommit` as in Java |
| `prepare_commit` | `prepareCommit()` / `prepareCommitInternal` / `startCommit` | **rewritten onto the real protocol** (F-1); double-prepare now rejected (F-3) |
| `finish_commit` | `finishCommit()` / `SegmentInfos.finishCommit` | **rewritten onto the real protocol** (F-1) |
| `rollback` | `rollback()` / `rollbackInternal` | divergent: does not close the writer or release a lock (there is none); now also deletes `pending_segments_N` (F-1). Does **not** delete orphaned segment files (F-11) |
| `apply_merge` | `commitMerge` / `checkpoint` | divergent: caller drives the merge; one bug fixed (F-4) |
| `auto_merge` | `maybeMerge` + `ConcurrentMergeScheduler` | divergent: synchronous, inside `finish_commit`. Owner: b10 |
| `execute_merge` | `merge(OneMerge)` | b10 |
| `segment_stats` | `MergePolicy.SegmentSizeAndDocs` | b10 |
| `set_live_commit_data` / `live_commit_data` | `setLiveCommitData` / `getLiveCommitData` | **newly ported** (F-6) |
| `segment_infos` | `getSegmentInfos` (package-private in Java) | identical |
| `pending_doc_count` | `numRamDocs` / `getNumBufferedDocuments` | identical |
| `committed_doc_count` | — | not-in-Java (Java has `numDocs`/`maxDoc`, live-aware; this is total-not-live and documented as such) |
| `next_segment_name` | `newSegmentName()` | divergent until F-2: Java bumps the counter as it hands out the name |
| `set_postings_field` / `add_postings_field` / `resolve_postings_field` | `FieldType.setIndexOptions` on the document side | not-in-Java (per-writer opt-in replaces per-field `FieldType`) |
| `set_custom_freq_postings_field` | — | not-in-Java |
| `set_term_vector_field` / `add_term_vector_field` / `resolve_term_vector_field` | `FieldType.setStoreTermVectors` | not-in-Java |
| `set_doc_values_field` | `Field` subclasses (`NumericDocValuesField`, …) | not-in-Java |
| `set_norms_field` | — | not-in-Java; Java writes norms for *every* indexed non-`omitNorms` field automatically (F-15) |
| `set_merge_policy` | `IndexWriterConfig.setMergePolicy` | b10 |
| `build_postings_output` | `FreqProxTermsWriter.flush` | divergent memory shape (F-9) |
| `build_custom_freq_postings_output` | — | not-in-Java |
| `build_norms_output` | `NormValuesWriter.flush` + `Similarity.computeNorm` | identical value (`SmallFloat.intToByte4(length)`); source pass now shared (F-8) |
| `build_term_vectors_output` | `TermVectorsConsumer.flush` | divergent: re-derived from the shared invert pass rather than accumulated during it |
| `build_*_doc_values_output` | `DocValuesWriter.flush` per type | divergent: single field only; b6 owns the codec side |
| `write_postings_files` / `write_doc_values_files` / `write_term_vector_files` / `write_norms_files` | `SegmentWriteState` + per-format `fieldsConsumer` | divergent: each re-reads and rewrites the `.si` to append its files, where Java accumulates `SegmentInfo.files` before writing the `.si` once (F-16) |
| `fields_with_per_field_attributes` | `PerFieldPostingsFormat.fieldsConsumer` / `PerFieldDocValuesFormat` | ported |
| `invert_pending_fields` | `IndexingChain.processField`'s single-invert fan-out | **new, from F-8** |
| `empty_segment_infos` | `new SegmentInfos(Version.LATEST.major)` | identical |
| `generate_segment_id` | `StringHelper.randomId()` | divergent (documented: no CSPRNG dependency; distinctness is the only property used) |
| `write_file` / `field_value_kind` / `to_segment_infos_version` | — | not-in-Java glue |

### Java `IndexWriter` methods with no Rust counterpart

`close`, `getConfig`, `getDirectory`, `getAnalyzer`, `flush`,
`flushNextBuffer`, `maybeMerge`, `forceMerge`, `forceMergeDeletes`,
`hasPendingMerges`, `getMergingSegments`, `hasDeletions`,
`hasUncommittedChanges`, `numDocs`, `maxDoc`, `getSegmentCount`,
`getFieldNames`, `ramBytesUsed`, `getFlushingBytes`, `numDeletedDocs`,
`advanceSegmentInfosVersion`, `advanceSegmentInfosCounter`,
`getSegmentInfosCounter`, `addDocuments` (block/`hasBlocks`), `addBatch`,
`updateDocuments`, `softUpdateDocument(s)`, `deleteDocuments(Query...)`,
`updateNumericDocValue`, `updateBinaryDocValue`, `updateDocValues`,
`tryDeleteDocument`, `tryUpdateDocValue`, `addIndexes(Directory...)`,
`addIndexes(CodecReader...)`, `getReader` (NRT), `IndexFileDeleter` and the
whole `tragedy` mechanism. Individually recorded under F-7/F-10/F-11/F-12.

---

## Findings

### F-1 `[CORRECTNESS]` — no `pending_segments_N`: a crash mid-commit made the index unopenable

**Java**: `SegmentInfos.prepareCommit(dir)` calls `dir.syncMetaData()` then
`write(dir)`, which serializes the commit to
`pending_segments_N`, closes it, fsyncs it, and deletes it if any of that
throws. `SegmentInfos.finishCommit(dir)` then `dir.rename(pending, segments_N)`
followed by `dir.syncMetaData()`. `rollbackCommit(dir)` deletes the pending
file. The name `pending_segments` deliberately does not start with `segments`,
so `getLastCommitGeneration` cannot see it.

**We did**: `segment_infos::write` created `segments_N` directly and fsynced
it. `IndexWriter::prepare_commit` did not write anything at all — it stashed a
`SegmentInfos` in a private field, and the module doc said so
("explicitly **not** crash-safe").

**Consequence**: two distinct failures.
1. A crash (or ENOSPC, or a partial write) between `create_output("segments_N")`
   and the fsync leaves a truncated `segments_N` at the *highest* generation.
   `last_commit_generation` picks it, `segment_infos::parse` rejects it, and
   **every** subsequent `IndexWriter::open` / `read_latest` fails — a
   recoverable crash turns the whole index unopenable. This got strictly worse
   with the (correct) tightening of `last_commit_generation` to be strict.
2. `prepare_commit` provided no durability at all: the "prepared" state was one
   in-memory field, so a crash lost it silently.

**Resolution — fixed.**
- `lucene-store`: `Directory::{rename, delete_file, sync_meta_data}` added
  (implemented for `FsDirectory` and `MmapDirectory` through shared
  `index_output` helpers) plus `pending_segments_file_name`
  (`IndexFileNames.fileNameFromGeneration(PENDING_SEGMENTS, …)`, including
  Java's `gen == -1 -> null` / `gen == 0 -> bare name` shape).
- `segment_infos.rs`: `write_pending` / `finish_pending` / `rollback_pending`,
  with `write` now being the first two back to back — so `update_document`,
  `delete_documents` and `apply_merge` became crash-safe without any call-site
  change.
- `index_writer.rs`: `prepare_commit` writes and fsyncs the pending file;
  `finish_commit` renames it and fsyncs the directory, and on failure **puts
  the prepared state back** rather than dropping it (the old `take()` would
  have orphaned an entire flushed segment on a failed publish); `rollback`
  deletes the pending file, ignoring errors exactly as Java's
  `IOUtils.deleteFilesIgnoringExceptions` does.

**Tests**:
`index_writer::tests::prepare_commit_writes_a_pending_segments_file_that_finish_commit_renames`
(asserts `pending_segments_2` exists and `segments_2` does *not* after prepare,
that a fresh reader still sees the previous commit — the crash case — and that
finish renames rather than copies),
`rollback_deletes_the_pending_segments_file_prepare_commit_wrote`, and in
`lucene-store`: `pending_segments_file_name_matches_file_name_from_generation`,
`a_pending_segments_file_is_invisible_to_the_commit_generation_scan`,
`rename_publishes_a_file_under_a_new_name_and_delete_file_removes_it`,
`rename_and_delete_file_surface_io_errors_for_a_missing_source`. In
`segment_infos.rs`: `write_pending_alone_does_not_publish_a_commit`, plus a
`FailingDir` test double (delegates to `FsDirectory`, fails `sync` or
`sync_meta_data` on demand) driving
`a_pending_commit_file_that_cannot_be_synced_is_deleted_not_left_behind` and
`a_renamed_commit_file_whose_directory_cannot_be_synced_is_deleted` — the two
`IOUtils.deleteFilesSuppressingExceptions` paths Java has and that no real
`FsDirectory` can reach on its own.

### F-2 `[CORRECTNESS]` — `update_document` never persisted the bumped segment counter

**Java**: `newSegmentName()` returns `"_" + counter++` and calls
`segmentInfos.changed()` in the same synchronized block, precisely so that
"we could close, re-open and re-return the same segment name that was
previously returned" cannot happen.

**We did**: `IndexWriter::update_document` computed the name from
`self.segment_infos.counter`, passed the *unbumped* `SegmentInfos` into
`update_document::update_document` (which cloned it and wrote it as the new
`segments_N`), and only afterwards did `self.segment_infos.counter += 1` on
the in-memory copy.

**Consequence**: the committed `segments_N` carried the pre-flush counter. A
writer reopened on that directory handed out the *same* `_N`, and its next
flush truncated (via `create_output`) the `.fdt`/`.fdx`/`.fdm`/`.fnm`/`.si` of
a segment the current commit still referenced — silent data loss and a corrupt
commit. Only reachable across a writer reopen, which is why no existing test
caught it.

**Resolution — fixed**: `update_document` now bumps `counter` on the
`SegmentInfos` that is actually written (and stamps a fresh commit id there,
see F-4), and the redundant in-memory bump is gone.

**Test**:
`update_document_persists_the_bumped_segment_counter_so_a_reopen_never_reuses_a_name`
— reopens the writer, asserts `next_segment_name()` differs from the name
`update_document` used, commits a third segment, and reads the replacement
document back to prove its files were not overwritten.

### F-3 `[CORRECTNESS]` — a second `prepare_commit()` silently discarded the first prepare's documents

**Java**: `prepareCommitInternal` throws
`IllegalStateException("prepareCommit was already called with no corresponding
call to commit")`.

**We did**: overwrote `self.prepared_commit` with a `SegmentInfos` rebuilt from
`self.segment_infos` — which had never seen the first prepare's segment.

**Consequence**: every document flushed by the first prepare was dropped from
the commit. The existing test
`calling_prepare_commit_again_before_finish_commit_replaces_the_pending_prepared_state`
asserted exactly this loss (it expected `["b"]` after preparing `"a"` then
`"b"`), so the bug was encoded as intended behaviour.

**Resolution — fixed**: `Error::PrepareCommitAlreadyCalled`. Test replaced with
`calling_prepare_commit_again_before_finish_commit_is_rejected_and_loses_nothing`,
which asserts the error, that the first prepare stays activatable, that the
document buffered after it is still buffered, and that both documents are
present after the following commit.

### F-4 `[CORRECTNESS]` — commit-writing operations run during a prepared commit silently reverted it

**Java**: every path that writes a `segments_N` is serialized on `commitLock`,
and nothing except `finishCommit` writes one while a `pendingCommit` exists.
Deletes and merges issued after `prepareCommit` are buffered into the *next*
commit.

**We did**: `delete_documents`, `apply_merge` and `update_document` each build
`self.segment_infos.clone()` with `generation + 1` and write it — the exact
generation a pending prepared commit has already claimed.

**Consequence**: `finish_commit` afterwards wrote `segments_N` at that same
generation from a segment list built *before* the delete/merge/update, silently
reverting it. On the new pending-file protocol it would also overwrite the
prepared pending file.

**Resolution — fixed**: `Error::PreparedCommitPending(op)` from
`update_document`, `delete_documents`, `apply_merge` and `delete_all` while
`prepared_commit.is_some()`. This is the analogue of Java's refusal to re-enter
`prepareCommit`: these operations *are* commits here, so the honest answer is
to make the caller finish or roll back first.

**Test**: `commit_writing_operations_are_refused_while_a_commit_is_prepared`,
which also proves the prepared commit survives the refusals intact.

### F-5 `[CORRECTNESS]` — every commit generation carried the same commit id

**Java**: `SegmentInfos.write(IndexOutput)` writes
`CodecUtil.writeIndexHeader(out, "segments", VERSION_CURRENT,
StringHelper.randomId(), …)` — a **fresh** id per commit file.

**We did**: wrote `segment_infos.id`, which every commit path inherited by
cloning the previous commit's `SegmentInfos`. Only `empty_segment_infos` ever
generated one, at `open`.

**Consequence**: every generation of an index reported the same
`SegmentInfos.getId()`, so nothing keyed on commit identity (segment
replication, "is this the same commit point") could distinguish two commits of
the same index.

**Resolution — fixed**: `prepare_commit`, `delete_documents`, `apply_merge` and
`update_document` all stamp `generate_segment_id(...)` onto the `SegmentInfos`
they write. Test: `every_commit_generation_gets_its_own_commit_id`.

### F-6 `[MISSING]` — `deleteAll()` and `setLiveCommitData`/`getLiveCommitData`

**Java**: `deleteAll()` aborts the buffered documents and clears
`segmentInfos` in memory (it does **not** write a commit; the emptiness becomes
durable at the next commit, and `IndexFileDeleter.checkpoint` reclaims the
files). `setLiveCommitData`/`getLiveCommitData` read and write
`SegmentInfos.userData`, the opaque per-commit metadata map — the mechanism
OpenSearch uses to tie a commit to a translog generation.

**We did**: neither existed. `SegmentInfos::user_data` was parsed and written
but completely unreachable from `IndexWriter`.

**Resolution — fixed**: `delete_all()` (same in-memory-only semantics as Java's,
including *not* deleting the dropped segments' files, which Java delegates to
the deleter this port does not have) and
`set_live_commit_data`/`live_commit_data`. Tests:
`delete_all_drops_buffered_and_committed_segments_but_only_on_next_commit`
(asserts a fresh reader still sees the pre-`deleteAll` commit until the next
`commit()`), `live_commit_data_is_written_into_the_commit_and_survives_reopen`.

### F-7 `[MISSING]` — no `DocumentsWriterDeleteQueue`, no sequence numbers

**Java**: every mutating method returns a sequence number from
`DocumentsWriterDeleteQueue`, and `updateDocument` is atomic precisely because
the delete and the add take adjacent slots in that queue: the delete applies to
every document with a smaller sequence number and to no document with a larger
one, and `FrozenBufferedUpdates` carries `delGen` so a later flush cannot apply
a delete to a document added after it.

**We do**: `add_document` returns `()`. `update_document`/`delete_documents`
resolve the delete eagerly against caller-supplied segments, then commit. The
ordering guarantee we actually provide is weaker but self-consistent: a delete
sees exactly the segments that exist at the moment it is called, and never the
writer's own unflushed buffer. `update_document` writes one `segments_N` after
both halves succeed, so no reader can observe a half-applied update — the
atomicity claim in `update_document.rs`'s module doc holds.

**Resolution — recorded, not fixed.** Porting sequence numbers means porting the
delete queue, `BufferedUpdates`, `FrozenBufferedUpdates` and
`BufferedUpdatesStream`, i.e. buffered (rather than eager) deletes, which is a
milestone of its own and changes every mutating signature. Recorded as a
carry-over.

### F-8 `[PERF]` — every consumer re-tokenized the same text (fixed, 1.74x)

**Java**: `IndexingChain.processField` calls `PerField.invert()` **once** per
`(document, field)` and fans the single result out to
`FreqProxTermsWriterPerField` (postings), `TermVectorsConsumerPerField` (term
vectors) and the `NormValuesWriter` (`FieldInvertState.length`).

**We did**: `build_postings_output`, `build_term_vectors_output` and
`build_norms_output` each built their own `(doc_id, field, text)` triples and
called `invert_documents` themselves. A commit with postings **and** norms on
one field analyzed and inverted every document twice; with term vectors as
well, three times. The norms pass in particular threw away everything except a
per-document token count.

**Measured** (`benchmarks/rust-runner`, `index-bench`, 20k docs x 40 tokens,
postings + norms on one field, `--release`, 3 runs each, fresh output
directory per run):

| configuration | ns/doc |
|---|---|
| postings + norms (baseline) | 45022 / 43813 / 43373 |
| postings only (`set_norms_field` skipped) | 26145 / 25860 / 25976 |

i.e. the redundant norms pass alone cost ~17.6 us/doc — **40% of the entire
commit** — to recompute a number the postings pass already had.

**Resolution — fixed**: new `IndexWriter::invert_pending_fields` inverts the
*union* of the postings, term-vector and norms fields exactly once per commit
(de-duplicated by field number, so a field opted into two consumers is still
analyzed once), and the three builders take that shared
`&InMemoryInvertedIndex`. `build_postings_output` walks its field's contiguous
`BTreeMap` range; `build_norms_output` and `build_term_vectors_output` filter
by field name. After the fix, the same benchmark: **25367 / 25214 / 24684
ns/doc** — norms became essentially free (1.74x on this shape). Every existing
postings/term-vector/norms/doc-values test still passes unchanged, which is the
correctness argument: the output is bit-identical, only the number of passes
changed.

### F-9 `[PERF]` — `BTreeMap` term dictionary during accumulation (fixed, further 1.6x)

**Java**: `TermsHashPerField` accumulates through a `BytesRefHash` —
open-addressed, hash-keyed, O(1) per occurrence — and the term dictionary is
*sorted* exactly once, at flush (`sortTerms()`). The postings themselves live
in `ByteBlockPool`/`IntBlockPool` slices grown through `ByteSlicePool`'s
tiered `LEVEL_SIZE_ARRAY` (5, 14, 20, 30, 40, 40, 80, 80, 120, 200 bytes), so a
term's posting stream costs no per-term `Vec` and no reallocation copy.

**We did**: `invert_documents` inserted into a `BTreeMap<(String, String),
Vec<PostingEntry>>` on **every** `(document, term)` occurrence group — an
O(log n) chain of `String` comparisons per touch, ~800k of them in the
benchmark, plus a per-document `BTreeMap<String, Vec<Occurrence>>` allocated
fresh per document.

**Resolution — partially fixed**: `invert_documents` now accumulates into a
`HashMap`, reuses one per-document grouping map across the whole batch
(`clear()`/`drain()` instead of a fresh allocation per document), and builds
the ordered `BTreeMap` once at the end from a sorted `Vec` (`BTreeMap`'s
`FromIterator` over sorted, deduplicated pairs builds bottom-up in O(n)). The
public `InMemoryInvertedIndex.terms` type and its ordering guarantee are
unchanged; the hash map's arbitrary iteration order is fully undone by the
existing sort-by-`doc_id` (each `(field, term)` receives at most one entry per
document, so `doc_id` totally orders the list) plus the new key sort.

**Measured** (A/B on the same binary, 5 runs each, with the F-8 fix in place in
both arms; the machine was under concurrent compile load, so the two high
outliers in the "after" arm are noise):

| | ns/doc |
|---|---|
| `BTreeMap` accumulation | 27918 / 27934 / 27853 / 29480 / 28985 |
| `HashMap` accumulation | 16217 / 17177 / 23731 / 30003 / 17950 |

and a quiet 5-run set after the change: 17474 / 17235 / 17397 / 17771 / 16626.

**Net over the batch**: ~43.9 us/doc -> ~17.4 us/doc, **~2.5x**, on the shape
`index-bench` measures.

**Still divergent, recorded**: the block-pool design itself. We still pay one
`String` per token (from `Analyzer::analyze`, which returns owned `Token`s —
`lucene-analysis`, b8), one `(String, String)` key per `(document, term)`
group, one `Vec<Occurrence>` per posting entry and one `Vec<PostingEntry>` per
term, where Java pays zero heap objects per occurrence. Closing that means
replacing `InMemoryInvertedIndex` with a `BytesRefHash`-equivalent over byte
pools and giving `Analyzer` a borrowed-token API — a redesign of two crates'
public types, not a contained fix. Carry-over.

### F-10 `[PERF]` / `[MISSING]` — no RAM accounting, no flush trigger, unbounded peak memory

**Java**: each `DocumentsWriterPerThread` owns a `Counter bytesUsed` fed by the
block pools and the per-field writers; `DocumentsWriterFlushControl` +
`FlushByRamOrCountsPolicy` flush a DWPT when `ramBufferSizeMB` (default
**16 MB**) or `maxBufferedDocs` (default `DISABLE_AUTO_FLUSH = -1`) is
exceeded, and `DocumentsWriterStallControl` back-pressures indexing threads
while too many bytes are in flight. Peak memory is therefore bounded by
configuration, independent of how many documents the caller adds between
commits.

**We do**: nothing. There is no byte counter anywhere in this crate,
`ramBufferSizeMB`/`maxBufferedDocs` have no equivalent, and `add_document`
cannot flush (it returns `()` and takes no `Directory` decision). Peak memory
is O(everything added since the last commit), with, at commit time,
simultaneously live:

1. every buffered `Document` (owned `String`s, kept until after the flush),
2. the shared `InMemoryInvertedIndex` — for the benchmark's shape, 20k docs x
   40 terms = ~800k `PostingEntry`s at ~44 bytes each plus a `Vec<Occurrence>`
   allocation each, ~35 MB before counting keys,
3. the `Vec<TermPostings>` copy handed to `postings_writer::write_fields`
   (`term: Vec<u8>`, `docs: Vec<(i32,i32)>`, `positions: Vec<Vec<i32>>`), and
4. every output file as a complete in-memory `Vec<u8>` before it is written.

So roughly three to four full copies of a segment, with no ceiling, where Java
holds one bounded 16 MB working set.

**Resolution — recorded, not fixed.** This is the honest headline memory
divergence of the write path. A `maxBufferedDocs`-style trigger is the cheap
half (it needs `add_document` to become fallible and to be able to flush), but
a real `ramBufferSizeMB` needs the block-pool redesign in F-9 to have anything
meaningful to measure — a byte counter over `String`/`Vec` allocations would be
an estimate of the wrong quantity. Both are recorded as carry-overs, and the
`IndexWriter` module doc's "No RAM-based flush triggering" note stays accurate.

### F-11 `[MISSING]` — no `IndexFileDeleter`: aborts and rollbacks leave every file behind

**Java**: `IndexFileDeleter` reference-counts every file named by every live
commit. `checkpoint(segmentInfos, isCommit)` incRefs the new commit and
decRefs the previous one, deleting anything that reaches zero;
`DocumentsWriterPerThread.abort()` deletes the files the aborted flush created;
`rollbackInternal` calls `deleter.refresh()`; `deleteAll` calls
`deleter.checkpoint`. Java also honours an `IndexDeletionPolicy` over which
older commits stay alive.

**We do**: nothing is ever deleted. Concretely, orphans accumulate from: a
`rollback()` after a `prepare_commit()` that already flushed a segment (the
`.fdt`/`.fdx`/`.fdm`/`.fnm`/`.si` and any postings/TV/DV/norms files stay);
`update_document`/`delete_documents` failing partway (the `.liv` files already
written stay — `update_document.rs`'s module doc already says so); every
superseded `segments_N` generation; every source segment of a merge; and every
segment dropped by `delete_all`. All of them are inert — no commit references
them — so this is a disk-space and `listAll()` -noise problem, not a
correctness one, and it is what makes the rest of the commit protocol
crash-safe in the first place (an orphan is always the safe outcome).

**Resolution — recorded, not fixed.** An `IndexFileDeleter` needs
`SegmentCommitInfo.files()` (per-segment file enumeration across `.liv` and
DV-update generations), an `IndexDeletionPolicy` equivalent, and a decision
about who owns deletion during a merge — it is a milestone, not a batch fix.
`Directory::delete_file` now exists, so the primitive is no longer the blocker.
Carry-over.

### F-12 `[MISSING]` — `softUpdateDocument`, `updateDocValues`, `deleteDocuments(Query)`, block adds

**Java**: `softUpdateDocument(Term, doc, Field... softDeletes)` adds the
document and applies a *doc-values update* (not a `.liv` bit) to the previous
one, which is how soft deletes and `SoftDeletesRetentionMergePolicy` work.
`updateNumericDocValue`/`updateBinaryDocValue`/`updateDocValues` are the same
mechanism directly. `deleteDocuments(Query...)` buffers a query-based delete.
`addDocuments`/`updateDocuments` add a contiguous block and set
`SegmentInfo.hasBlocks`.

**We have**: none of them. `flush_stored_only_segment` hardcodes
`has_blocks: false`, and there is no doc-values *update* write path
(`lucene-codecs`'s `doc_values_updates.rs` is read-side; b6 owns it).
`lucene-search`'s `soft_deletes.rs` reads soft deletes but nothing writes them.

**Resolution — recorded, not fixed.** All four need the buffered-updates
machinery from F-7 and/or a doc-values-update writer. Carry-over.

### F-13 `[CORRECTNESS]` (port-wide, minor) — `SegmentCommitInfo` id always written as absent

**Java**: every freshly flushed or merged segment gets
`new SegmentCommitInfo(..., StringHelper.randomId())`
(`DocumentsWriterPerThread` line 507, `IndexWriter` lines 3550/5086), so
`SegmentInfos.write` emits marker byte `1` plus 16 bytes. `sciId == null` only
happens for indices written before 7.4.

**We do**: every `SegmentCommitInfo` constructed anywhere in this port sets
`sci_id: None` (marker `0`) — `segment_writer.rs`, `update_document.rs`,
`merge.rs`, `term_delete.rs`, `points_delete.rs`, `check_index.rs`,
`directory_reader.rs`. `deletes.rs` correctly propagates whatever it was given.

**Consequence**: none for readability — `SegmentInfos.readCommit` accepts
marker `0` without complaint, and nothing in Lucene validates the id. What is
lost is per-segment-commit identity for anything that keys on it (segment
replication dedup).

**Resolution — recorded, not fixed.** Fixing only `segment_writer.rs` would
make the port internally inconsistent; this wants one change across all the
`SegmentCommitInfo` construction sites, which span b9/b10/b11 and
`lucene-search`. Carry-over, with the note that a deterministic derivation from
`segment_id` satisfies the only property anything uses (distinctness), so no
CSPRNG dependency is needed.

### F-14 `[MISSING]` — no position/offset validation in the invert pass

**Java**: `IndexingChain.PerField.invert` throws on a first position increment
of 0, a negative increment, position overflow past `Integer.MAX_VALUE`,
`startOffset < lastStartOffset` ("offsets must not go backwards"), and
`endOffset < startOffset`.

**We do**: `invert_documents` accumulates `position += token.position_increment`
with no checks, so a 0 first increment would produce position `-1`.

**Consequence**: currently unreachable. `invert_documents` takes a concrete
`&Analyzer` (`lucene-analysis` exposes a struct, not a trait), and every filter
in that pipeline emits `position_increment >= 1` and monotonic offsets, so no
caller can inject a violating token stream. It becomes reachable the moment
`Analyzer` becomes an extension point.

**Resolution — recorded, not fixed** (an unreachable branch would also be
uncoverable under the 95% line bar). Flagged for whoever makes `Analyzer`
pluggable.

### F-15 `[MISSING]` — norms are opt-in per field; Java writes them for every indexed field

**Java**: `IndexingChain` creates a `NormValuesWriter` for every field whose
`IndexOptions != NONE` and `omitNorms == false`, with no opt-in.

**We do**: `set_norms_field` opts in exactly one field, and
`fields_with_per_field_attributes` force-sets `omit_norms = true` on every
other indexed field so the `.fnm` does not promise norms the segment lacks
(commit `a4a812f`). That keeps real Lucene able to open the segment — omitting
norms is a legal configuration — but it silently overrides what the caller
declared in its `FieldInfo`, and BM25 then scores that field with a constant
norm.

**Resolution — recorded, not fixed.** The blocker is on the codec side:
`norms::write_single_dense_field` is single-field-only, so writing norms for
every indexed field needs a multi-field `.nvd`/`.nvm` writer. `norms.rs` is
b6's file and is being edited concurrently. The `IndexWriter` half is now
essentially free — F-8's shared invert pass already has every field's
per-document length. Carry-over, owner b6 + a follow-up here.

### F-16 `[PERF]` — the `.si` is read back, re-parsed and rewritten once per file group

**Java**: `SegmentInfo.files` is accumulated in memory
(`SegmentInfo.setFiles`/`addFiles`) while each format writes, and
`SegmentInfoFormat.write` runs once at the end of `sealFlushedSegment`.

**We do**: `flush_stored_only_segment` writes the `.si`, then
`write_postings_files`, `write_term_vector_files`, `write_doc_values_files` and
`write_norms_files` each `dir.open(".si")`, `segment_info::parse` (full parse
including checksum verification), extend `files`, `segment_info::write`, write
and fsync it again. A commit with postings + term vectors + doc values + norms
does that five times.

**Consequence**: four redundant read-parse-write-fsync cycles per commit. The
`.si` is small (hundreds of bytes), so the cost is four extra fsyncs, not
bandwidth — real but fixed-per-commit, and invisible next to F-8/F-9 in the
benchmark (which amortizes it over 20k documents).

**Resolution — recorded, not fixed.** The contained fix is to thread the file
list through `flush_stored_only_segment` (or split it into
"write-codec-files" / "write-si") rather than patching after the fact — a
signature change to a function `update_document.rs`, `merge.rs`, both fixture
examples and the FFI all call. Not worth doing while `merge.rs` is being edited
by b10. Carry-over.

### F-17 `[INTENTIONAL]` — `commit()` always writes a new generation

Java's `prepareCommitInternal`/`startCommit` skip the whole commit when
`pendingCommitChangeCount == lastCommitChangeCount` ("no changes pending"). We
have no `changeCount`, so an empty `commit()` writes the next `segments_N` with
`version` bumped and no new segment. Deliberate and documented on `commit()`;
the result is a valid commit, just a redundant one. Adding a change counter is
cheap but changes the return contract (`commit()` returning the *previous*
generation), so it belongs with the `hasUncommittedChanges` work.

### F-18 `[INTENTIONAL]` — no `IndexWriterConfig`, no write lock, no `close()`

`open` takes four parameters instead of an `IndexWriterConfig`; there is no
`NativeFSLockFactory` write lock, no `close()`, no `tragedy` state, and
`rollback()` leaves the writer usable (Java's makes it permanently closed).
All documented in the module doc. Single-caller, sequential-use scope; revisit
when concurrency arrives.

### F-19 `[INTENTIONAL]` — postings and custom-freq postings are mutually exclusive per writer

`set_custom_freq_postings_field` and `set_postings_field` cannot both be active
(`Error::PostingsAndCustomFreqPostingsMutuallyExclusive`). Real Lucene mixes
`IndexOptions` freely across fields in one segment; the restriction is a scope
cut, documented at the method, and the two builders would need to union their
`FieldPostingsInput`s into one `write_fields` call to lift it.

### F-20 `[PERF]` — impacts are still computed against norm 1

Carried over from b5 and re-verified here: `postings_writer` emits one impact
`(maxFreq, norm = 1)` per block, where
`Lucene104PostingsWriter` feeds real per-document norms into a
`CompetitiveImpactAccumulator` and emits the Pareto-optimal `(freq, norm)`
set. Sound (norm 1 is the highest-scoring norm, so the bound is valid) but
loose, costing block-level pruning.

**Status change**: the `lucene-index` half of the blocker is gone — F-8's
shared invert pass already computes every document's field length in the same
pass that builds the postings, so the norms are available at the
`write_fields` call site. What remains is a `lucene-codecs` change (a norms
input on `FieldPostingsInput` and a real `CompetitiveImpactAccumulator`), which
is not safe to make while three batches are editing that crate. Carry-over,
owner: whoever next owns `postings_writer.rs`.

### Answer to the question raised alongside this batch

`last_commit_generation` was the only lenient generation/commit-file handler,
and it is now strict. Re-checked every other site in these files: `open` is the
only caller of the generation scan; `committed_doc_count`, `segment_stats`,
`write_postings_files`/`write_doc_values_files`/`write_term_vector_files`/
`write_norms_files` (`.si` read-back) and `flush_sorted_stored_only_segment`
all propagate `Err` and never substitute a default. The only deliberate
error-swallowing introduced by this batch is in `write_pending`'s cleanup
deletion and `rollback_pending`, both of which mirror Java's
`IOUtils.deleteFilesSuppressingExceptions`/`deleteFilesIgnoringExceptions`
exactly.

---

## Verdicts

- **`lib.rs`** — swept clean.
- **`indexing_chain.rs`** — swept; F-9 fixed (measured), F-14 recorded
  (unreachable), block-pool redesign recorded as a carry-over.
- **`segment_writer.rs`** — swept; no fix needed in this file. Open: F-11
  (abort leaves files), F-13 (`sci_id`), F-16 (`.si` rewrite) — all
  cross-batch.
- **`update_document.rs`** — swept; F-2's fix lands in its `IndexWriter`
  caller (the function itself is unchanged and correct given its documented
  contract). Open: F-7 (delete queue / sequence numbers).
- **`index_writer.rs`** — swept; F-1 through F-6 and F-8 fixed with tests.
  Open: F-7, F-10, F-11, F-12, F-15, F-16, F-20.

## Summary

20 findings:

- **CORRECTNESS 6** — F-1 (no `pending_segments_N`), F-2 (unpersisted segment
  counter), F-3 (double prepare loses documents), F-4 (commit-writing ops
  revert a prepared commit), F-5 (one commit id for every generation) all
  **fixed with tests**; F-13 (`sci_id` always absent) recorded — it needs one
  change across every `SegmentCommitInfo` construction site in the port.
- **MISSING 6** — F-6 (`deleteAll`, `setLiveCommitData`/`getLiveCommitData`)
  **fixed with tests**; F-7 (delete queue / sequence numbers), F-11
  (`IndexFileDeleter`), F-12 (`softUpdateDocument`/`updateDocValues`/
  `deleteDocuments(Query)`/block adds), F-14 (invert-pass validation,
  unreachable today), F-15 (norms are opt-in) recorded with the blocker for
  each.
- **PERF 5** — F-8 (one invert pass per consumer) and F-9 (`BTreeMap`
  accumulation) **fixed and benchmarked**, 43.9 -> 17.4 us/doc (~2.5x) on
  `index-bench`; F-10 (no RAM accounting / flush trigger), F-16 (`.si`
  rewritten per file group), F-20 (impacts against norm 1) recorded.
- **INTENTIONAL 4** — F-17 (`commit()` always writes a generation), F-18 (no
  `IndexWriterConfig`/lock/`close`), F-19 (postings paths mutually exclusive),
  plus the `generate_segment_id` non-CSPRNG choice already documented in the
  code.

All carry-overs are recorded in `docs/sweep/m2/LEDGER.md`.
