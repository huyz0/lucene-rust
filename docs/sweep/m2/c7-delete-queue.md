# c7-delete-queue

Follow-up batch opened from one blocker recorded three times: b9's F-7 (no
`DocumentsWriterDeleteQueue`, no sequence numbers), which b9 itself recorded as
blocking four separate public APIs (F-12), b6 recorded as blocking
`BinaryDocValuesFieldUpdates`, and c3 recorded as sharing a blast radius with
`inflateGens`' per-segment half (c3 F-3) and b9's F-13 (`sci_id`).

Files swept, and the Java counterparts compared against (all under
`/home/tuong/work/lucene/lucene/core/src/java/org/apache/lucene/index/` unless
stated):

| Rust file | Java counterpart(s) |
|---|---|
| `crates/lucene-index/src/buffered_updates.rs` (**new**) | `DocumentsWriterDeleteQueue.java`, `BufferedUpdates.java`, `FrozenBufferedUpdates.java`, `DocValuesUpdate.java`, `BufferedUpdatesStream.java`, `FieldUpdatesBuffer.java` |
| `crates/lucene-index/src/index_writer.rs` | `IndexWriter.java`, `DocumentsWriter.java`, `DocumentsWriterPerThread.java`, `FrozenBufferedUpdates.java` (the `apply*` half), `ReadersAndUpdates.java` |
| `crates/lucene-index/src/segment_infos.rs` | `SegmentCommitInfo.java` |
| `crates/lucene-index/src/index_file_deleter.rs` | `IndexFileDeleter.inflateGens`, `IndexFileNames.parseGeneration` (`.../index/IndexFileNames.java`) |
| `crates/lucene-index/src/segment_writer.rs` | `DocumentsWriterPerThread.flush`/`sealFlushedSegment`, `SegmentInfo.setHasBlocks` |
| `crates/lucene-index/src/deletes.rs` | `SegmentCommitInfo.advanceDelGen`, `ReadersAndUpdates.writeLiveDocs` |
| `crates/lucene-codecs/src/doc_values_updates.rs` | `BinaryDocValuesFieldUpdates.java`, `DocValuesFieldUpdates.java` |
| `crates/lucene-ffi/src/writer.rs` | `IndexWriter.updateDocument`/`deleteDocuments(Term...)` (the visibility contract only) |
| `crates/lucene-codecs/src/norms.rs` | `Lucene90NormsProducer.readFields` (**decision only, file not modified** -- see F-19) |

---

## `crates/lucene-index/src/buffered_updates.rs` (new)

The port of the sequence-number machinery itself.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `DeleteQueue::next_sequence_number` | `DocumentsWriterDeleteQueue.getNextSequenceNumber()` | identical (starts at 1, `+1` per operation) |
| `DeleteQueue::last_sequence_number` | `getLastSequenceNumber()` | identical |
| `DeleteQueue::skip_sequence_numbers` | `skipSequenceNumbers(long)` | identical |
| `DeleteQueue::add_term_deletes` | `addDelete(Term...)` + `TermArrayNode.apply` | identical semantics; slice/global split done eagerly (F-2) |
| `DeleteQueue::add_query_deletes` | `addDelete(Query...)` + `QueryArrayNode.apply` | as above |
| `DeleteQueue::add_doc_values_updates` | `addDocValuesUpdates(DocValuesUpdate...)` + `DocValuesUpdatesNode.apply` | as above |
| `DeleteQueue::freeze_global_buffer` | `freezeGlobalBuffer(DeleteSlice)` + `globalBufferedUpdates.clear()` | identical outcome |
| `DeleteQueue::freeze_private_buffer` | `DocumentsWriterPerThread.prepareFlush`/`flush`'s `pendingUpdates` handling | divergent on *when* term deletes are resolved, identical in outcome (F-3) |
| `DeleteQueue::any_changes` | `anyChanges()` | identical |
| `DeleteQueue::clear` | `clear()` | identical, and the seqNo counter deliberately survives |
| `BufferedUpdates::add_term` | `addTerm(Term, int)` | identical, including "keep the higher `docIDUpto`" |
| `BufferedUpdates::add_query` | `addQuery(Query, int)` | identical (`Map.put` overwrite semantics) |
| `BufferedUpdates::add_doc_values_update` | `addNumericUpdate` + `addBinaryUpdate` | identical; one method because `DocValuesUpdate` already carries the discriminant |
| `BufferedUpdates::{any,clear,clear_delete_terms}` | `any()`, `clear()`, `clearDeleteTerms()` | identical |
| `FrozenBufferedUpdates::new` | `new FrozenBufferedUpdates(...)` | identical content; `PrefixCodedTerms` replaced by a sorted `Vec` (F-4) |
| `FrozenBufferedUpdates::{del_gen,set_del_gen,any}` | `delGen()`, `setDelGen(long)`, `any()` | identical |
| `FrozenBufferedUpdates::applies_to` | the `if (segState.delGen > delGen) continue;` guard repeated in `applyTermDeletes`/`applyQueryDeletes`/`applyDocValuesUpdates` | identical rule, extracted to one place |
| `FrozenBufferedUpdates::limit_for` | the `if (delGen == segState.delGen) limit = …; else limit = Integer.MAX_VALUE;` branch, same three methods | identical rule, extracted to one place |
| `BufferedUpdatesStream::{push,next_gen,pending,any,clear}` | `push`, `getNextGen`, `updates`, `any`, `clear` | identical |
| `BufferedUpdatesStream::clear_applied` | `FinishedSegments.finishedSegment` + `waitApply` | collapsed (F-1) |
| `Term`, `DocValuesUpdate`, `FieldUpdatesBuffer`, `BufferedUpdate`, `UpdateValue` | `Term`, `DocValuesUpdate.{Numeric,Binary}DocValuesUpdate`, `FieldUpdatesBuffer`, `FieldUpdatesBuffer.BufferedUpdate` | identical fields |
| `DeleteQuery` | `org.apache.lucene.search.Query` | **not-in-Java** by dependency-graph necessity (F-5) |
| — | `DeleteSlice`, `Node`/`TermNode`/`QueryNode`/…, `globalBufferLock`, `tryApplyGlobalSlice`, `updateSlice`, `advanceQueue`, `getMaxCompletedSeqNo`, `isAdvanced`, `close` | **not ported**, all concurrency plumbing (F-1) |
| — | `BufferedUpdatesStream.{waitApplyAll,waitApplyForMerge,stillRunning,getCompletedDelGen}`, `FinishedSegments` | **not ported**, same reason (F-1) |
| — | `Accountable`/`ramBytesUsed` on all four classes, `BYTES_PER_DEL_QUERY`, `DeletedTerms`' `ByteBlockPool`+`BytesRefIntMap` | **not ported** (F-6) |

### Findings

#### F-1 `[INTENTIONAL]` — the lock-free linked list and the slice heads are not ported

**Java**: `DocumentsWriterDeleteQueue` is a singly linked list of `Node`s with a
`volatile` tail, one `DeleteSlice` head per `DocumentsWriterPerThread` plus a
global slice, a `ReentrantLock` around the global buffer, and
`BufferedUpdatesStream.FinishedSegments` tracking which `delGen`s are still
being resolved by other threads. The class doc states the purpose outright: "a
non-blocking linked pending deletes queue" — the structure exists so that many
indexing threads can append deletes without a lock and still agree on ordering.

**We do**: keep the two things the slices actually *compute* — a private
`BufferedUpdates` for the segment being built and a global one for the segments
already written — and drop the list, the heads, the locks and the in-flight
tracking. This port has one indexing thread by construction (`index_writer.rs`'s
module doc: "one caller, one `Directory`, sequential calls").

**Equivalence argument** (this is the part worth being explicit about, since it
is the whole justification): Java applies a slice *lazily*, at the next
`finishDocuments(deleteNode, docsInRamBefore)`, with `docIDUpto =
docsInRamBefore`. Walk the two cases with one thread:

- A standalone `deleteDocuments(t)` issued when the RAM buffer holds *K*
  documents. Java buffers the node; the next `finishDocuments` call passes
  `docsInRamBefore == K` (the add that follows has not been counted yet), so the
  term lands at `docIDUpto = K`. If no further add happens, `prepareFlush`
  applies it at `numDocsInRAM == K`. Either way: *K*. We record *K* at issue
  time.
- `updateDocument(t, doc)` when the buffer holds *K*. Java indexes the document
  first (`numDocsInRAM` becomes *K+1*), then adds the delete node and applies it
  at `docIdUpTo = docsInRamBefore == K` — so the delete reaches docs `0..K` and
  not the new one. We record *K* before buffering the document.

The two are the same number in both cases, so eager and lazy application are
observationally identical for one thread. `docs/sweep/m2` has no
multi-threaded `IndexWriter` to make them differ.

**Resolution — deliberate, recorded.** Restoring the list is what a
multi-threaded `DocumentsWriterPerThreadPool` would need, and that is a
milestone, not a batch item. The one thing kept from the concurrent design is
`skip_sequence_numbers`, so the seqNo space still shows a commit as a
discontinuity.

#### F-2 `[INTENTIONAL]` — one delete goes into both buffers at issue time

**Java**: a node is appended once; the private slice reads it with the DWPT's
`docIDUpto` and the global slice reads it with `BufferedUpdates.MAX_INT`.

**We do**: `add_term_deletes`/`add_query_deletes`/`add_doc_values_updates` write
the entry into both buffers directly, with `doc_id_upto` and
`MAX_DOC_ID_UPTO` respectively. Same two results, no shared node to walk.

#### F-3 `[PERF]` — a freshly flushed segment's own term deletes cost a `.liv` generation this port writes and Java does not

**Java**: `FreqProxTermsWriter.applyDeletes` resolves the DWPT's private *term*
deletes against the in-RAM `FreqProxFields` **before** the postings are written,
folds the matches into `SegmentWriteState.liveDocs`, and then calls
`pendingUpdates.clearDeleteTerms()`. The new segment is therefore born with its
own deletes already inside its live-docs bitset, and no `.liv` file exists at
generation 1.

**We do**: keep the term deletes in the segment's private packet and resolve
them from the *written* segment, through the same path every other segment's
deletes take. The outcome is identical — the `docIDUpto` limit is what bounds
them either way, and the test
`a_delete_issued_before_a_flush_still_reaches_the_segment_that_flush_produces`
pins it — but the segment gains a `_N_1.liv` where Java has none.

**Cost**: one extra file per flush *that had buffered deletes*, which is
`live_docs::write` over a `max_doc`-bit bitset (a few hundred bytes to a few
kilobytes) plus one fsync. Zero when nothing was deleted, which is the common
case. Not fixed: folding it into the flush needs the delete terms resolved
against `InMemoryInvertedIndex` rather than the blocktree, i.e. a second
resolution path with its own correctness surface, to save one small file.
**Recorded**, with the note in `freeze_private_buffer`'s own doc comment.

#### F-4 `[PERF]` — `PrefixCodedTerms` is a sorted `Vec`, not a prefix-coded byte block

**Java**: `FrozenBufferedUpdates` compresses its delete terms into
`PrefixCodedTerms` and its class doc measures the win at ~8.3% of the original
size, because a packet can be held in memory across many segments' resolution
while other threads work.

**We do**: a `Vec<(Term, i32)>` sorted by `(field, bytes)`. Sorted for the same
reason Java prefix-codes sorted — a term-dictionary walk is cheapest in term
order — but uncompressed.

**Consequence**: a packet's residency here is one call stack (F-1: it is pushed
and applied inside the same `flush()`), so the memory it holds is transient
rather than pooled across threads, and 8.3% of a transient structure is not
worth a second byte-block encoding. **Recorded, not fixed.**

#### F-5 `[INTENTIONAL]` — `DeleteQuery` is a closed enum, not `lucene_search::Query`

**Java**: `deleteDocuments(Query...)` takes `org.apache.lucene.search.Query` and
`FrozenBufferedUpdates.applyQueryDeletes` resolves it with a real
`IndexSearcher` over the segment reader.

**We cannot**: the dependency graph is strictly downward (`util ← store ←
codecs ← index ← search ← core ← ffi`) and `lucene-search` depends on
`lucene-index`, so naming a `lucene_search::Query` here inverts the edge into a
cycle. `term_delete.rs` and `points_delete.rs` already document exactly this
constraint for their own resolve halves.

**We do**: `DeleteQuery` is a closed enum of the shapes this crate can resolve
with the primitives it owns — a blocktree term dictionary and a postings
reader: `Term`, `Prefix`, `TermRange`, `MatchAll`, `Any`, `All`, `Not`.
`lucene-search`/`lucene-ffi` can lower their richer `Query` onto it; anything
outside the set stays a caller-side resolution, exactly as it was before this
module existed. Resolution lives in `index_writer.rs` (see that file's F-11).

#### F-6 `[INTENTIONAL]` — no `Accountable`/`ramBytesUsed` on the buffers

**Java**: all four classes implement `Accountable`, and
`DocumentsWriterFlushControl` uses the delete queue's byte count as one of its
flush triggers (`getApplyAllDeletes`).

**We do**: nothing. c3's flush trigger measures the buffered-document arena
(c3 F-8 records why that is a different quantity from Java's), and adding a
byte count over `HashMap<Term, i32>` would be an estimate of a third quantity
again. The buffers are bounded by the number of *operations* between flushes,
which the document-count and RAM triggers already bound indirectly.
**Recorded.**

### Verdict

New file, swept against all six Java classes. No CORRECTNESS or MISSING items
open. Open: F-3 (one extra `.liv` per flush-with-deletes, measured cost stated),
F-4 (no prefix coding), F-6 (no `ramBytesUsed`) — all recorded with reasons.
16 unit tests.

---

## `crates/lucene-index/src/index_writer.rs`

### Method correspondence (only what this batch changed or added)

| Rust | Java | Verdict |
|---|---|---|
| `add_document` | `addDocument(Iterable)` | **now returns `Result<SeqNo>`** (F-7) |
| `add_documents` | `addDocuments(Iterable<Iterable>)` | **newly ported** (F-10) |
| `update_document` | `updateDocument(Term, doc)` | **rewritten onto buffered semantics + seqNo** (F-8); the old eager form kept as `update_document_with_sources` |
| `update_documents` | `updateDocuments(Term, Iterable<Iterable>)` | **newly ported** (F-10) |
| `delete_documents_by_term` | `deleteDocuments(Term...)` | **rewritten onto buffered semantics + seqNo** (F-8) |
| `delete_documents_by_query` | `deleteDocuments(Query...)` incl. the LUCENE-6379 `MatchAllDocsQuery` specialisation | **newly ported** (F-11) |
| `soft_update_document` | `softUpdateDocument(Term, doc, Field...)` | **newly ported** (F-12) |
| `soft_update_documents` | `softUpdateDocuments(Term, Iterable<Iterable>, Field...)` | **newly ported** (F-12) |
| `update_doc_values` | `updateDocValues(Term, Field...)` + `buildDocValuesUpdate` | **newly ported** (F-13) |
| `update_numeric_doc_value` | `updateNumericDocValue(Term, String, long)` | **newly ported** (F-13) |
| `update_binary_doc_value` | `updateBinaryDocValue(Term, String, BytesRef)` | **newly ported** (F-13) |
| `verify_doc_values_update_field` | `globalFieldNumberMap.verifyOrCreateDvOnlyField` | divergent: verifies, never creates (F-13) |
| `add_document_with_custom_freq_terms` | — | not-in-Java; now returns a seqNo for consistency |
| `add_documents_with_delete` | `IndexWriter.updateDocuments(Node, docs)` + `DocumentsWriterPerThread.updateDocuments`/`finishDocuments` | the shared body; ordering is the atomicity guarantee (F-8) |
| `flush` | `DocumentsWriterPerThread.flush` + `IndexWriter.publishFlushedSegment` | now freezes/pushes the two packets and stamps `bufferedDeletesGen` (F-9) |
| `apply_all_deletes_and_updates` | `IndexWriter.applyAllDeletesAndUpdates` + `BufferedUpdatesStream.waitApplyAll` | **newly ported** (F-9) |
| `apply_packets_to_segment` | `FrozenBufferedUpdates.apply(SegmentState[])` + `applyTermDeletes`/`applyQueryDeletes`/`applyDocValuesUpdates` | ported, in Java's three-phase order (F-25); one open per segment instead of one per packet (F-14) |
| `open_segment_for_deletes` | `ReaderPool.get` / `ReadersAndUpdates`' reader half | **newly ported**, scoped to what a delete needs (F-14) |
| `resolve_delete_query` | `applyQueryDeletes`' `weight.scorer(readerContext)` walk | ported for `DeleteQuery`'s shapes (F-11) |
| `resolve_term_span` | `PrefixQuery`/`TermRangeQuery`'s `AutomatonQuery` term enumeration | divergent shape, same span (F-11) |
| `write_doc_values_update_generation` | `ReadersAndUpdates.writeFieldUpdates` + `SegmentCommitInfo.advanceDocValuesGen` | ported semantics, **not** Lucene's bytes (F-15) |
| `rollback` | `rollbackInternal` | now also `deleteQueue.clear()` + `bufferedUpdatesStream.clear()` (F-16) and `segmentInfos.rollbackSegmentInfos(rollbackSegments)` (F-24) |
| `rollback_segments` field | `IndexWriter.rollbackSegments` + `SegmentInfos.createBackupSegmentInfos` | **newly ported** (F-24) |
| `delete_all` | `deleteAll()` | now also clears the queue and the stream (F-16) |
| `update_document_with_sources` / `delete_documents_with_sources` | — | not-in-Java; this port's pre-existing eager primitives, kept (F-8) |
| — | `tryDeleteDocument`, `tryUpdateDocValue`, `updateDocuments(Query, ColumnBatch)`, `addIndexes`, `getReader`/NRT, `forceMerge`, `IndexWriterConfig.getIndexSortFields` validation | **still missing**, unchanged by this batch |

### Findings

#### F-7 `[MISSING]` — no sequence numbers (b9 F-7). **Fixed.**

**Java**: every mutating `IndexWriter` method returns a `long` sequence number
"showing the effective serialization of all operations".

**We did**: `add_document` returned `Result<()>`;
`update_document`/`delete_documents` returned the new `SegmentInfos`.

**Fixed**: every mutating method returns `Result<SeqNo>`, sourced from
`DeleteQueue::next_sequence_number`, starting at 1 exactly as Java's does
("seqNo must start at 1 because some APIs negate this to also return a
boolean"). Tests:
`every_mutating_method_returns_a_strictly_increasing_sequence_number` (walks all
six entry points and asserts strict monotonicity plus `seqs[0] == 1`),
`a_document_block_takes_exactly_one_sequence_number` (a three-document block
consumes one number, not three — it is one operation),
`a_rollback_never_reissues_a_sequence_number`.

Note the return-type change is source-compatible for the `?;`/`.unwrap();`
statement call sites that existed, which is why no caller outside this crate
needed a change for it.

#### F-8 `[MISSING]` — deletes were eager and immediately committed (b9 F-7). **Fixed.**

**Java**: `updateDocument`/`deleteDocuments(Term...)` **buffer**. The delete
takes a slot in the delete queue adjacent to the add, and the pair is atomic
precisely because of that adjacency.

**We did**: both resolved the term against caller-supplied `SegmentDeleteSource`s
and wrote a `segments_N` immediately — one commit generation per updated
document.

**Fixed**: `update_document(Term, doc)` and `delete_documents_by_term(&[Term])`
buffer and return a seqNo; the change lands at the next flush/commit. The
ordering inside `add_documents_with_delete` *is* the guarantee: the delete's
`docIDUpto` is read **before** the call's documents are buffered
(Java's `finishDocuments(deleteNode, docsInRamBefore)`), so it reaches every
document that already existed and none being added — the replacement included.

The old eager forms are kept, renamed `update_document_with_sources` /
`delete_documents_with_sources`, because they cover a case the buffered path
cannot: a segment whose postings this writer cannot open itself (a checked-in
real-Lucene fixture whose `.tim`/`.tip`/`.tmd` live outside the writer's
directory). Their tests are unchanged.

Tests — the ones that actually pin the contract rather than the method's
existence:
- `interleaved_adds_updates_and_deletes_produce_the_expected_visible_set`: the
  five-operation ordering `add a / add b / update(alpha→a2) / add c /
  delete(beta)` inside one segment produces exactly `["a2", "c"]`.
- `a_delete_does_not_reach_a_document_added_after_it_in_the_same_segment`: a
  delete issued at buffer position 1 leaves the identically-termed document
  added afterwards alive.
- `a_delete_issued_before_any_document_exists_deletes_nothing`.
- `two_updates_of_the_same_term_in_one_buffer_leave_only_the_newest`, which is
  what `BufferedUpdates.addTerm`'s "keep the higher `docIDUpto`" rule buys.
- `update_document_replaces_across_a_flush_boundary`,
  `a_delete_issued_after_a_commit_reaches_the_committed_segment`,
  `a_rollback_discards_buffered_deletes_along_with_buffered_documents`.

#### F-9 `[MISSING]` — no `delGen`, so nothing decided which segments a delete could reach. **Fixed.**

**Java**: `BufferedUpdatesStream.push` stamps each frozen packet with the next
generation; `publishFlushedSegment` gives the new segment
`bufferedDeletesGen = nextGen` *after* publishing the global packet; and every
`apply*` method skips a segment whose `delGen` exceeds the packet's. A delete
therefore reaches the segments that existed when it was issued and no others.

**We did**: nothing — an eager delete simply saw whatever segments existed at
call time, which was self-consistent but had no equivalent for a *buffered*
one.

**Fixed**: `flush()` now, in Java's order,
1. freezes the global buffer and pushes it — **before** the new segment takes a
   generation, which is exactly what keeps the new segment out of the packets
   that predate it;
2. writes the segment;
3. freezes the private buffer and pushes it (or burns a generation via
   `next_gen()` when there is nothing to push, as `publishFlushedSegment` does),
   and stamps the result on the segment's `buffered_deletes_gen`;
4. calls `apply_all_deletes_and_updates`, which walks the pending packets oldest
   generation first and applies each to every segment `applies_to` admits.

This is the half that is easy to get subtly wrong and hard to notice, so it has
the sharpest test:
`a_delete_applies_to_the_segments_flushed_before_it_and_not_to_the_ones_after`
flushes `_0` with two documents carrying term `shared`, issues
`delete_documents_by_term(shared)`, then flushes `_1` with two *more* documents
carrying the same term, and asserts the visible set is exactly `["c", "d"]`
with `segments[0].del_count == 2` and `segments[1].del_count == 0`. Its mirror,
`a_delete_issued_before_a_flush_still_reaches_the_segment_that_flush_produces`,
asserts the opposite direction. `buffered_updates.rs`'s
`a_packet_applies_to_older_segments_and_not_to_newer_ones` and
`only_a_segments_own_private_packet_honours_the_doc_id_upto_limit` pin the two
rules in isolation.

#### F-10 `[MISSING]` — no block adds, `hasBlocks` hardcoded `false` (b9 F-12, b10). **Fixed.**

**Java**: `addDocuments`/`updateDocuments` add a run of documents guaranteed to
occupy contiguous ascending doc IDs, take **one** sequence number for the whole
run, and set `SegmentInfo.hasBlocks` when the run held more than one document
(`DocumentsWriterPerThread.updateDocuments`: `if (numDocs > 1)
segmentInfo.setHasBlocks()`). It is what makes parent/child join queries legal
against the segment.

**We had**: neither method, and `flush_stored_only_segment` hardcoded
`has_blocks: false`.

**Fixed**: `add_documents`/`update_documents`/`soft_update_documents` on
`IndexWriter`; `segment_writer::flush_stored_only_segment_with_blocks` as a new
entry point (a *sibling* rather than a ninth parameter, because Java's is a
mutator on an already-built `SegmentInfo` and because all ~25 existing call
sites of the old function mean `false` — several of them in crates other batches
hold). Contiguity is guaranteed by consulting the flush threshold **once** per
call, after the whole block is buffered, exactly as Java's `doAfterDocument`
runs once per `updateDocuments` rather than once per document.

Tests: `a_block_add_sets_has_blocks_on_the_flushed_segment`,
`single_document_adds_leave_has_blocks_unset` (including that a one-document
`add_documents` is *not* a block, matching `numDocs > 1`),
`has_blocks_does_not_leak_from_one_flush_into_the_next`,
`an_automatic_flush_never_splits_a_document_block` (`max_buffered_docs = 2`, a
four-document block, one segment out),
`update_documents_deletes_the_old_block_and_adds_the_new_one_atomically`.

**And against real Lucene**: a new
`crates/lucene-index/examples/write_block_segment_fixture.rs` +
`fixtures/src/VerifyBlockSegment.java` case in
`scripts/verify-write-path.sh` writes 300 blocks of 4 documents and has real
Lucene assert that `LeafMetaData.hasBlocks()` is **true**, that the parent
postings land at doc IDs `0, 4, 8, …` (the contiguity `hasBlocks` promises),
and that `CheckIndex` at `MIN_LEVEL_FOR_SLOW_CHECKS` is clean. Deliberately no
index sort and no parent field: Lucene only *requires* a parent field for a
block-carrying segment when an index sort is present (`CheckIndex.testSort`,
`IndexWriter.mergeMiddle`'s `hasBlocksButNoParentField`), and the unsorted case
is the one this port can produce today.

#### F-11 `[MISSING]` — no `deleteDocuments(Query...)` (b9 F-12). **Fixed.**

**Java**: buffers the queries and resolves each against a segment at apply time
with a real `IndexSearcher`, honouring `deleteQueryLimits[i]` when the packet is
that segment's own.

**Fixed**: `delete_documents_by_query(&[DeleteQuery])`, resolving against the
segment's blocktree/postings. LUCENE-6379's specialisation is ported: a
`MatchAllDocsQuery` anywhere in the array short-circuits to `delete_all()`,
which drops whole segments instead of writing an all-zero `.liv`.
`Prefix`/`TermRange` walk the term dictionary and **stop** at the end of the
span (the dictionary is sorted, so nothing after it can match), which is what
keeps a prefix delete proportional to the matching span rather than to the
field.

Tests: one per shape —
`delete_documents_by_query_resolves_a_term_query`,
`…_a_prefix_and_stops_at_the_span_end`,
`…_an_inclusive_and_an_exclusive_range`,
`…_boolean_composition` (`All`),
`…_a_negation_over_live_docs_only` (`Not`),
`…_a_union` (`Any`),
`…_specialises_match_all_into_delete_all`, plus
`a_query_delete_honours_the_doc_id_upto_limit_within_its_own_segment`.

See `buffered_updates.rs`'s F-5 for why the query type is a closed enum.

#### F-12 `[MISSING]` — no `softUpdateDocument` (b9 F-12). **Fixed.**

**Java**: adds the document and, instead of deleting the ones matching `term`,
applies `softDeletes` to them as doc-values updates — the whole soft-delete
mechanism. `buildDocValuesUpdate(term, softDeletes)` then routes them through
the very same `updateDocuments(Node, docs)` path a hard update uses, so add and
marking share one sequence number.

**Fixed**: `soft_update_document`/`soft_update_documents`, with Java's
"at least one soft delete must be present" check ported as
`Error::NoSoftDeletesSupplied`. Test
`soft_update_document_adds_the_new_doc_and_marks_the_old_one_without_deleting_it`
asserts all three properties that make it a *soft* delete: nothing anywhere has
`del_count > 0`, both versions are still readable, and the marking is present on
the original and **absent on the replacement** — because the doc-values update
carries the buffer position it was issued at, exactly like a hard delete would.

#### F-13 `[MISSING]` — no `updateDocValues`/`updateNumericDocValue`/`updateBinaryDocValue` (b9 F-12). **Fixed.**

**Fixed**: all three, plus the `buildDocValuesUpdate` retargeting (every
supplied field becomes an update keyed by the caller's term) and Java's
validation. One divergence, deliberate: Java's `verifyOrCreateDvOnlyField`
*creates* an absent field; this port's field list is fixed at
`IndexWriter::open`, so an unknown field is `Error::UnknownDocValuesUpdateField`
rather than an implicit schema change, and a type mismatch is
`Error::WrongDocValuesUpdateType`.

A `None` value is Java's "if a doc values fields data is `null` the existing
value is removed from all documents matching the term", which reaches
`DocValuesFieldUpdates.reset(doc)` — **not** a write of zero. b6 added the
`Option<i64>` representation that expresses it; this batch is the first caller
that can reach it from the public API. Tests:
`update_numeric_doc_value_writes_a_generation_the_reader_can_replay`,
`update_doc_values_with_a_null_value_records_a_removal_not_a_zero`,
`update_binary_doc_value_writes_a_binary_generation`,
`successive_doc_values_updates_accumulate_as_separate_generations` (ascending
generation order, newest last — the order
`numeric_value_with_generations` expects),
`update_doc_values_rejects_an_unknown_or_wrongly_typed_field`,
`a_doc_values_update_generation_is_referenced_and_survives_the_deleters_sweep`.

#### F-14 `[PERF]` / `[INTENTIONAL]` — one segment open per apply pass, not one per packet

**Java**: resolves each packet through a pooled `ReadersAndUpdates`, so the
segment reader is opened once and reused across every packet, merge and NRT
reopen for as long as the pool holds it.

**We do**: `open_segment_for_deletes` reads the segment's `.si` (for `max_doc`),
its current `.liv`, and its `.tim`/`.tip`/`.tmd`/`.doc` — the slice of
`ReadersAndUpdates` a delete actually needs — once per `apply_packets_to_segment`
call, and resolves *all* applicable packets against that one open before writing
one `.liv` generation and one doc-values generation. Same outcome (deletes are a
set union; doc-values updates are applied in ascending generation order so the
newest still wins), one open and one generation bump instead of N.

Against Java this is worse across many apply passes (no pool: the next
flush reopens) and better within one (N packets, one open). A reader pool is
the same milestone as NRT `getReader`, which this port does not have.
**Recorded.**

#### F-15 `[CORRECTNESS]` (scoped, recorded) — a segment carrying doc-values updates is not readable by real Lucene

**Java**: `ReadersAndUpdates.writeFieldUpdates` rewrites the field through
`Lucene90DocValuesConsumer` into a generational `.dvd`/`.dvm` pair named by
`IndexFileNames.fileNameFromGeneration`, and records it in
`SegmentCommitInfo.getDocValuesUpdatesFiles()`.

**We do**: write one overlay file per `(generation, field)` in
`doc_values_updates.rs`'s format — **this port's own invention**, as that
module's doc comment has said since b6 — named
`<segment>_<gen>_LuceneRustDVU_<fieldNumber>.dvu`. The name is deliberately
shaped as Java's four-part `segment_gen_codec_suffix.ext` so that
`IndexFileNames.parseGeneration` (and this port's port of it, see
`index_file_deleter.rs` below) reads the generation back out of it, and the
files are recorded in `dv_update_files` so `SegmentCommitInfo::files()` — and
therefore the deleter, `check_index` and `checksum_verify` — see them.

**Consequence, stated plainly**: an index *with* doc-values updates cannot be
handed to real Lucene, because `CheckIndex` and any doc-values read of that
field would try to open the recorded files as `Lucene90DocValuesFormat` output.
An index *without* them still can, which is why `verify-write-path.sh`'s cases
are unaffected (none of them writes a doc-values update) and still 17/17.

**Not fixed**: closing it means a generational-`.dvd`/`.dvm` writer plus the
`docValuesGen` file-naming contract — the same work b6 scoped out of
`doc_values_updates.rs` in the first place. This batch's contribution is that
the *semantics* above it (which docs, which value, which generation wins, reset
vs zero) are now ported and tested, so the remaining work is the byte format
alone. **Carry-over**, added to the ledger.

#### F-16 `[CORRECTNESS]` — `rollback`/`deleteAll` would have leaked buffered deletes into the next segment. **Fixed as written.**

**Java**: `rollbackInternal` closes the delete queue and calls
`bufferedUpdatesStream.clear()`; `deleteAll` calls
`docWriter.lockAndAbortAll()` (which clears the queue) and clears the stream.

**We do**: the same, in both methods. Without it, a delete buffered before a
rollback would apply to whatever was added *after* it — its `docIDUpto` indexes
a document buffer that no longer exists. The seqNo counter deliberately does
**not** reset (Java builds a fresh queue whose numbers continue past the aborted
ones), so a caller can never see the same seqNo twice from one writer. Tests:
`a_rollback_discards_buffered_deletes_along_with_buffered_documents`,
`a_rollback_never_reissues_a_sequence_number`.

#### F-25 `[CORRECTNESS]` — a doc-values update reached a document the same packet had just deleted. **Fixed.**

**Java**: `FrozenBufferedUpdates.apply` runs `applyTermDeletes` →
`applyQueryDeletes` → `applyDocValuesUpdates` **in that order** inside one
packet, and the third reads `final Bits acceptDocs = segState.rld.getLiveDocs()`
— which by then already reflects the first two, because they called
`segState.rld.delete(docID)`. A document the packet just killed therefore takes
no doc-values update.

**We did**: filter update targets only against the segment's *pre-pass* live
docs, so a document deleted earlier in the same apply pass still received the
update.

**Consequence**: bounded — a dead document's doc-values value is not something
a correct reader consults, since it filters by live docs first. What it costs is
a wasted overlay entry and, more importantly, a divergence in what the file
*says*: a hard-deleted document could appear in a soft-delete overlay, which is
exactly the kind of contradictory state a retention policy reading both would
have to reason about.

**Fixed**: the accumulator is a `HashSet<i32>` and the update loop filters
`!deleted.contains(&d)`. Only the same packet's own deletes are visible to it,
matching Java: across packets Java iterates a `HashSet<FrozenBufferedUpdates>`,
so cross-packet ordering is not guaranteed there either — what governs which
update *wins* across generations is `delGen`, not apply order.

**Test**: `a_doc_values_update_skips_a_document_the_same_packet_just_deleted` —
two documents sharing a term, one hard-deleted by term and both targeted by a
numeric update in the same buffer; the deleted document's overlay entry must be
absent and the survivor's present. Verified to discriminate: it failed against
the pre-fix code with `left: Some(Some(9)), right: None`.

#### F-24 `[CORRECTNESS]` — a rollback after a buffered delete had been applied left the in-memory view pointing at a reclaimed file. **Fixed.**

**Java**: `IndexWriter` keeps `rollbackSegments`, a backup of the segment list
taken in the constructor and refreshed in `finishCommit`
(`pendingCommit.createBackupSegmentInfos()`); `rollbackInternal` restores it
with `segmentInfos.rollbackSegmentInfos(rollbackSegments)`.

**We did not have it**, and before this batch it did not matter: the committed
segment list was never mutated between commits, because the eager delete paths
wrote their own `segments_N` immediately. Buffered deletes changed that —
`apply_all_deletes_and_updates` bumps a *committed* segment's `del_gen` in
`self.segment_infos` and writes a `.liv` that no commit yet names. `rollback()`
then runs `deleter.refresh()`, whose whole job is to reclaim exactly such an
unreferenced file. The in-memory view survived the rollback pointing at a
deleted file, and the next `commit()` would write a `segments_N` naming it.

This is a defect the batch introduced and the batch's own review caught; it is
recorded rather than quietly fixed because the interaction — "buffered deletes
make a previously immutable structure mutable, which makes a previously
unnecessary snapshot necessary" — is the kind of thing the next person touching
this area needs to know.

**Fixed**: `rollback_segments` on `IndexWriter`, captured at `open` and
refreshed at all four sites that install a durable commit (`finish_commit`,
`update_document_with_sources`, `delete_documents_with_sources`, `apply_merge`
— all four are exactly the sites followed by `checkpoint_committed`), restored
in `rollback()`. `generation`/`version`/`counter` deliberately stay put, as in
Java: they only ever move forward.

`flush()`'s empty-buffer path gained the matching
`deleter.checkpoint(&live, false)` too — a delete issued with nothing buffered
writes its `.liv` through the early return, and without the checkpoint that
file was unreferenced until the next commit.

**Test**: `a_rollback_after_a_buffered_delete_was_applied_restores_the_committed_segment_list`
— commit one document, buffer a delete, `flush()` it (so the `.liv` is written
and `del_gen` is 1), roll back, then assert the view is back at `del_gen == -1`,
the orphaned `_0_1.liv` is gone from the directory, and every file the *next*
commit names actually exists. Verified to discriminate: with the restore line
removed the test fails.

#### F-17 `[PERF]` — `add_document` does not allocate a one-element `Vec`

The obvious implementation of `add_document` after F-10 is
`add_documents(vec![doc])`. That puts a heap allocation and a move in front of
the hot path of the entire write side, to reach a code path that can never set
`has_blocks` (`numDocs > 1`) and carries no delete node. `add_document` keeps
its own three-line body over the shared `buffer_document` helper instead.

### Verdict

Swept for the delete/update surface. F-7 through F-13, F-16, F-24 and F-25
fixed with tests; F-14 and F-17 are design notes with reasons; **F-15 is the one open
correctness-shaped item** and is a carry-over with a named blocker. b9's
F-15 (norms opt-in), F-16 (`.si` rewritten per file group), F-20 (impacts vs
norm 1) and F-9 (block-pool `indexing_chain`) are untouched by this batch and
unchanged.

---

## `crates/lucene-index/src/segment_infos.rs`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `SegmentCommitInfo.next_write_del_gen` field + `next_write_del_gen()` | `nextWriteDelGen` + `getNextDelGen()` | **newly ported** (F-18) |
| `next_write_field_infos_gen` / `next_write_doc_values_gen` (+ accessors) | `nextWriteFieldInfosGen` / `nextWriteDocValuesGen` (+ getters) | **newly ported** (F-18) |
| `set_next_write_*_gen` | `setNextWrite*Gen(long)` | identical |
| `advance_del_gen` / `advance_doc_values_gen` | `advanceDelGen()` / `advanceDocValuesGen()` | identical |
| `buffered_deletes_gen` + `set_buffered_deletes_gen` | `bufferedDeletesGen` + `setBufferedDeletesGen(long)` | **newly ported**, including the "only set while `-1`" guard (F-18) |
| `impl Default for SegmentCommitInfo` | the constructor's defaults | not-in-Java (Rust needs it for the sentinel; see F-18) |
| `sci_id` at construction | `StringHelper.randomId()` | **fixed** (F-13-of-b9, see `segment_writer.rs` below) |

### Findings

#### F-18 `[MISSING]` — `SegmentCommitInfo` had none of Java's four transient generation fields (c3 F-3, b9 F-13's blast radius). **Fixed.**

**Java**: `SegmentCommitInfo` carries `nextWriteDelGen`, `nextWriteFieldInfosGen`
and `nextWriteDocValuesGen` (constructor: `delGen == -1 ? 1 : delGen + 1`, and
`inflateGens` may raise them) plus `bufferedDeletesGen` (default `-1`). None of
the four is serialized into `segments_N`.

**We did**: none of them. `term_delete`/`deletes` derived the next `.liv`
generation as `del_gen + 1` inline, so there was nothing for `inflateGens` to
raise and nothing for a delete packet to compare against.

**Fixed**, all four, as `pub` fields with Java's semantics. Two design notes
worth recording because they are what made the change land without touching
other batches' files gratuitously:

- The three `next_write_*_gen` fields use `0` as a **"not explicitly set"
  sentinel** and the accessors derive Java's constructor value from `del_gen`/
  `field_infos_gen`/`doc_values_gen` when they hold it. `0` is never a legal
  Lucene generation, which is what makes it usable. The payoff is that the ~40
  existing struct literals across five files only needed
  `..Default::default()` appended rather than three correct-per-site values,
  and a parsed `SegmentCommitInfo` still round-trips `PartialEq` against a
  hand-built one.
- `buffered_deletes_gen` defaults to `-1`, so a segment read back from a commit
  is admitted by *every* packet a fresh writer session can issue — which is
  correct: everything in a commit predates every delete that session makes.

The literal-site fixups reached `merge.rs` and `lucene-search/directory_reader.rs`
(other batches' files); both are one added line inside an existing literal and
nothing else in those files was touched.

### Verdict

Swept for the generation bookkeeping. F-18 fixed. `SegmentInfos`' own parse/
write is b11's and is unchanged.

---

## `crates/lucene-index/src/index_file_deleter.rs`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `inflate_gens(files, &mut infos)` | `inflateGens(SegmentInfos, Collection<String>, InfoStream)` | **now complete** — was partial, see F-19 |
| `parse_generation(file_name)` | `IndexFileNames.parseGeneration(String)` | **newly ported** (F-19) |

### Findings

#### F-19 `[MISSING]` — `inflateGens`' per-segment half (c3 F-3). **Fixed.**

**Java**: after a crash the directory can hold files at higher generations than
the last commit records. `inflateGens` pushes `SegmentInfos.generation` and
`counter` past everything on disk (c3 ported this) **and** each segment's three
`nextWrite*Gen` past the highest generation seen in any of that segment's file
names, so a name a crashed session may already have written is never handed out
again.

**We did**: only the first half — c3 recorded the second as blocked on the
`SegmentCommitInfo` fields, which F-18 has now added.

**Fixed**: `inflate_gens` takes `&mut SegmentInfos` (Java mutates the
`SegmentInfos` it is given) and does both halves. Java's own comment about the
per-segment maximum is ported verbatim into the doc, because the behaviour is
otherwise surprising: the maximum is the **union** of the live-docs, field-infos
and doc-values generations, "since it means DV updates will suddenly write to
the next gen after live docs' gen, for example, but we don't have the APIs to
ask the codec which file is which". This port inherits that rather than
inventing a per-group split real Lucene does not have. Java only ever *raises*
each counter, and so does this.

`parse_generation` is Java's `IndexFileNames.parseGeneration` including its
four documented name shapes and its `catch (NumberFormatException)` — a
trash tail reads as "no generation" rather than erroring.

`deletes::apply_deletes` now derives its next `.liv` generation from
`sci.next_write_del_gen()` instead of `del_gen + 1`, which is what makes the
inflation actually take effect.

Tests: `inflate_gens_pushes_a_segments_next_write_gens_past_a_crashed_sessions_liv`
(the exact crash: the commit records `_0` at delGen 1 but a `_0_3.liv` from a
dead session is still on disk; the next delete must go to generation 4, not 2),
`inflate_gens_never_lowers_a_segments_next_write_gen`,
`inflate_gens_leaves_a_segment_with_no_files_on_disk_at_its_derived_gens`,
`parse_generation_reads_javas_four_file_name_shapes`.

### Verdict

Swept clean for `inflateGens`. c3's other two open items (`SnapshotDeletionPolicy`,
the `.si` parse in `commit_files`) are untouched and unchanged.

---

## `crates/lucene-index/src/segment_writer.rs`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `flush_stored_only_segment_with_blocks` | `DocumentsWriterPerThread.flush` + `SegmentInfo.setHasBlocks()` | **new** (F-10 above) |
| `flush_stored_only_segment` | same, `hasBlocks = false` | unchanged behaviour, now a wrapper |
| `derive_sci_id` | `StringHelper.randomId()` | **newly ported** (F-20) |

### Findings

#### F-20 `[CORRECTNESS]` (port-wide, minor) — `sci_id` was always written as absent (b9 F-13). **Fixed.**

**Java**: every freshly flushed (`DocumentsWriterPerThread`) or merged
(`IndexWriter.mergeMiddle`) segment gets `new SegmentCommitInfo(...,
StringHelper.randomId())`, so `SegmentInfos.write` emits marker byte `1` plus 16
bytes. `sciId == null` only happens for indices written before 7.4.

**We did**: every construction site in the port set `sci_id: None`.

**Fixed** at the two sites that *create* a segment — `segment_writer.rs` and
`merge.rs`'s two merged-segment constructors (a one-line change each; the rest
of `merge.rs` untouched). Every other site propagates whatever it was given and
needed nothing. As b9 predicted, no CSPRNG is needed: distinctness is the only
property anything reads the id for, and `derive_sci_id` mixes the segment's
already-unique id. Real Lucene reads it back without complaint —
`scripts/verify-write-path.sh` is 17/17 with every case now emitting marker
byte `1`.

### Verdict

Swept for this batch's scope. b9's F-11 (abort file cleanup) was closed by c3;
F-16 (`.si` rewritten per file group) is unchanged.

---

## `crates/lucene-codecs/src/doc_values_updates.rs`

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `write_binary_updates` | `BinaryDocValuesFieldUpdates.add(int, BytesRef)` + `DocValuesFieldUpdates.finish()`'s stable sort | **newly ported** (F-21) |
| `read_binary_updates` | `BinaryDocValuesFieldUpdates.iterator()` | **newly ported** (F-21) |
| `binary_value_with_updates` | `ReadersAndUpdates`' single-generation overlay read, BINARY | **newly ported** (F-21) |
| `binary_value_with_generations` | `DocValuesFieldUpdates.mergedIterator`, BINARY | **newly ported** (F-21) |
| the four numeric equivalents | the NUMERIC side | unchanged; b6's `reset(doc)` support built on, not undone |
| — | `PagedMutable`/`BytesRefBuilder` buffering, `Container`, `Accountable` | no Rust counterpart — Java-object-shape plumbing |

### Findings

#### F-21 `[MISSING]` — `BinaryDocValuesFieldUpdates` had no counterpart (b6 #2). **Fixed.**

**Java**: runs BINARY updates through the same `DocValuesFieldUpdates` base
class as NUMERIC and differs only in how a value is buffered (`BytesRefBuilder`
+ `PagedMutable` offsets/lengths instead of a `PagedMutable` of longs).

**Fixed**: the binary side of all four functions, with semantics identical to
the numeric side's — ascending by doc, last write per doc wins within a
generation, newest generation wins across them, and a `None` value is
`reset(doc)`. Two details worth calling out:

- **`Some(vec![])` and `None` are deliberately distinguishable.** An empty
  `BytesRef` is a legal `BinaryDocValues` value; a `reset` is the absence of
  one. Test:
  `binary_reset_round_trips_as_a_none_value_distinct_from_an_empty_one`.
- **A distinct codec name** (`LuceneRustBinaryDocValuesUpdates`), so handing a
  numeric overlay to the binary reader fails the header check instead of
  decoding garbage, and vice versa. Test:
  `binary_overlay_rejects_a_numeric_overlay_file`.

There is no `VERSION_START`-era binary format to stay compatible with — the
binary side did not exist before b6's `VERSION_HAS_VALUE` — so its `has_value`
byte is unconditional.

13 tests, including the corruption cases (wrong segment id, corrupt footer,
out-of-order doc ids, negative value length) and the generation-resolution
cases (newest wins whether it sets or resets; an older generation's reset
survives a disjoint newer one; no generations degenerates to the base decode).

b6's finding #3 (per-doc map lookup rather than a merged priority queue) is
unchanged and still INTENTIONAL for the same reason.

### Verdict

Swept; F-21 fixed with tests. b6's #3 intentional. The module's declared scope —
the file layout is this port's own invention — is unchanged, and F-15 above
records what that costs now that a writer actually emits these files.

---

## `crates/lucene-ffi/src/writer.rs`

Not a sweep of the FFI surface (that was b15/c6); this section exists because
the batch changed a semantic the JVM side observes, and the coordinator asked
for it to be stated explicitly.

### F-22 `[CORRECTNESS]` — the FFI delete path is now **buffered**, matching Java

**What it was**: `ffi_writer_update_document` and `ffi_writer_delete_documents`
reopened *every* committed segment from the writer's directory themselves
(`open_all_segment_sources`/`build_delete_sources`, ~160 lines in this crate)
and drove the eager `IndexWriter::update_document`/`delete_documents`, which
resolved and **committed immediately**. A JVM caller saw the change on disk the
moment the call returned, and paid one `segments_N` generation *per updated
document*.

**What it is now**: both delegate to the buffered
`IndexWriter::update_document(Term, doc)` /
`delete_documents_by_term(&[Term])`. The operation goes into the delete queue
and takes effect at the next `ffi_writer_commit` — exactly what
`IndexWriter.updateDocument`/`deleteDocuments(Term...)` do in Java, and exactly
the timing `ffi_writer_add_document` already had. The ~160 lines of segment
reopening are deleted from this crate: the wrapped `IndexWriter` owns it now.

**ABI unchanged** — both still take the same parameters and return `i32`. What
changed is *when* the effect is visible. The sequence number is not surfaced
across the boundary; adding it would need a new out-parameter, which is a
deliberate non-change here (no caller has asked for it, and doing it silently
would be an ABI break).

**Unchanged**: a segment with no `.tim` file on disk still contributes no
matches rather than erroring — there is simply no term dictionary to resolve
against.

**Tests**: the four end-to-end tests that asserted immediate visibility now
assert the *new* timing explicitly — read back after the call and confirm
nothing changed, then commit and confirm it did — rather than being quietly
retargeted. Plus one new test,
`a_buffered_delete_does_not_reach_a_document_added_after_it`, which is the
boundary-level statement of the whole batch: a delete issued after a commit
kills the committed document, and an identically-termed document added *after*
the delete survives.

### Verdict

Migrated, with the visibility change stated in the module doc, both function
docs and five tests.

---

## `crates/lucene-codecs/src/norms.rs` — decision, file not modified

### F-23 `[MISSING]` — `parse_meta` does not validate field numbers against `FieldInfos` (b6 #4). **Recorded again, deliberately not fixed.**

**Java**: `Lucene90NormsProducer.readFields` throws `CorruptIndexException` both
for an unknown field number and for a field whose `FieldInfo` says it has no
norms.

**We do**: `parse_meta` takes no `FieldInfos`, so a corrupt `.nvm` naming a
nonexistent field is accepted into `Norms.entries` and then never matched by an
`entry(field_number)` lookup. Missed diagnostic only; **no wrong value can
result**, because every caller looks up by a field number it got from its own
`FieldInfos`.

**Decision: not now.** Four reasons, in descending weight.

1. **c7 is not in fact touching this area.** The batch brief's premise was "now
   that you are touching this area". It turns out c7 does not: the doc-values
   work is in `doc_values_updates.rs` (a sibling module with no norms coupling)
   and the rest is the index write path. c7 adds **zero** new `parse_meta` call
   sites. Doing it here would be an unrelated change riding along in a batch
   whose diff is already large.
2. **The call sites are held by three other live batches right now.** The 22
   sites live in `lucene-index/{index_writer,merge,check_index}.rs`,
   `lucene-search/{lib,directory_reader}.rs` + three of its integration tests,
   `lucene-ffi/segment.rs`, `lucene-codecs/tests/norms_fixtures.rs` and
   `benchmarks/rust-runner`. `merge.rs` is c8's, `check_index.rs` is c9's,
   `lucene-search` is c11's. That is the *same collision* that made b6 defer it,
   not a new one, and the coordinator's standing instruction is not to touch
   another batch's in-flight files.
3. **The mechanical fix is the wrong shape.** Threading `FieldInfos` through
   `norms::parse_meta` alone leaves the port half-validated:
   `doc_values::parse_meta` already validates, `norms` would, and every other
   per-field meta parser still would not. A one-off is a local improvement that
   makes the inconsistency harder to see, not easier.
4. **There is a strictly better moment.** Every one of the 22 sites currently
   opens `.fnm` itself and then opens each per-format file separately. When the
   reader side gains a `SegmentReader`-equivalent that parses `.fnm` once and
   threads it into every per-format open — which `lucene-search`'s
   `SegmentReader` is already growing toward (b13 gave it `.nvm`/`.nvd`) — the
   `FieldInfos` is already in hand at the call site and the parameter costs
   nothing. Doing it before that means 22 mechanical edits that the later
   refactor rewrites anyway.

**Recorded** in `docs/sweep/m2/LEDGER.md` with this reasoning attached, so the
next batch that lands the shared-`FieldInfos` reader picks it up as a
one-line-per-site addition rather than rediscovering the tradeoff.

---

## Gates

- `cargo fmt --all` — clean.
- `cargo build --workspace` — clean.
- `cargo clippy -p lucene-index --all-targets -- -D warnings` — clean for this
  batch's files.
- `cargo test -p lucene-index -p lucene-ffi` — green: `lucene-index` 515 unit +
  16 integration, `lucene-ffi` 441, 0 failures. (One `check_index.rs` failure
  was visible mid-batch; it was c9's file mid-edit and is gone.)
- `scripts/verify-write-path.sh` — **17/17**, up from 16/16, with the new
  `VerifyBlockSegment` case. Every case now also exercises the `sci_id` fix
  (F-20): real Lucene reads marker byte `1` plus 16 bytes on every segment this
  port writes.

### Performance: what the sequence-number machinery costs per document

The brief asks specifically whether "a correctness feature that costs 20%
throughput" is what landed. It is not measurable.

**Method**: two binaries built from the identical tree, differing only in the
seqNo machinery — the A arm removes `next_sequence_number()` from
`add_document`, the global freeze/push from `flush()`, the private packet and
`set_buffered_deletes_gen`, and the `apply_all_deletes_and_updates()` call.
`index-bench`, 100k documents x 40 tokens, postings + norms on one field,
`--release`, fresh output directory per run, arms **interleaved** A/B/A/B so
machine-load drift cancels rather than accumulating in one arm. The machine was
running five other sweep batches throughout (load average ~5.6), which is why
the run-to-run spread is wide and why interleaving matters.

| pair | A (no seqNo machinery), ns/doc | B (this batch), ns/doc |
|---|---|---|
| 1 | 22148 | 23546 |
| 2 | 20848 | 19143 |
| 3 | 17803 | 16954 |
| 4 | 17100 | 17098 |
| 5 | 19192 | 19287 |
| 6 | 26906 | 26266 |
| 7 | 20430 | 20517 |
| 8 | 23007 | 23576 |
| **median** | **20639** | **19900** |
| **min** | **17100** | **16954** |

The sign of the pairwise difference alternates (B is nominally faster in 5 of
8 pairs, and faster on both the median and the minimum), which is the signature
of noise rather than of a real effect in either direction. **The seqNo
machinery's per-document cost is below this benchmark's noise floor**, and the
reason is structural rather than lucky:

- Per `add_document`, the machinery adds exactly one `i64` increment. F-17
  removed the one-element `Vec` that the obvious implementation would have put
  there.
- Per flush, `freeze_global_buffer` returns `None` and `freeze_private_buffer`
  returns `None` when nothing was buffered, so the only work is one integer
  bump (`next_gen()`), and `apply_all_deletes_and_updates` early-returns on an
  empty stream without opening anything.
- The expensive part — opening a segment and resolving terms — runs only when a
  delete or update actually exists, i.e. never on this benchmark's shape, which
  is exactly Java's cost model too.

Absolute throughput (median ~19.9 us/doc, min 16.95) is consistent with c3's
~21 us/doc baseline at 100k with the flush trigger on; the memory line is
unchanged (writer peak ~128-131 MB, 4 segments).

---

## Tier-2 review (`quality-reviewer`)

Run against this batch's files after the gate was first green. It confirmed the
part most at risk — "the freeze/push/stamp sequence in `flush()` matches
`publishFlushedSegment`", `applies_to`/`limit_for` against the
`segState.delGen > delGen` and `delGen == segState.delGen` branches, the
private-segment filter against `getInfosToApply`, the `0 == derive` sentinel,
and `parse_generation` against `IndexFileNames.parseGeneration` — and found
**three gating defects**, all real, all now fixed with tests verified to
discriminate.

#### F-26 `[CORRECTNESS]` — a rollback rewound the delete generation but not the segments, so every later delete reached only the oldest segment. **Fixed.**

**Java**: `BufferedUpdatesStream.clear()` does set `nextGen = 1` — and it is
safe there, because `rollbackInternal` then **closes** the writer. No delete is
ever issued again against the `SegmentCommitInfo`s the rollback restored.

**We did**: copy that line. This port deliberately leaves the writer usable
after `rollback()` (`index_writer.rs`'s doc comment states that divergence
explicitly), and F-24's restore hands back segments still carrying the
`buffered_deletes_gen` their original flush stamped on them — 1, 2, 3, … . The
next packet was then stamped generation 1, and `applies_to`
(`segment_gen <= del_gen`) rejected it for every segment above generation 1.

**Consequence**: a **silent, partial delete** after a rollback — the worst
shape a delete bug can take, since the call succeeds and the index looks fine.
Reproduced by the reviewer: two committed segments, rollback, delete a term
present in both, commit → one document survives and `del_count` is `[1, 0]`.

**Fixed**: `BufferedUpdatesStream::clear()` no longer resets `next_gen`. This
is the one place the port must diverge from Java's `clear()`, and it is the
same argument `DeleteQueue::clear` already makes for the sequence-number
counter: a generation, once handed out, must never be handed out again while
anything that saw it is still reachable. Documented at the method.

**Tests**: `a_delete_after_a_rollback_still_reaches_every_committed_segment`
(two committed segments — one is not enough, `gen 1 == gen 1` passes by luck,
which is exactly why the pre-existing rollback test missed it) and
`stream_clear_drops_the_packets_but_never_rewinds_the_generation`. Both verified
to fail with the reset restored.

#### F-27 `[CORRECTNESS]` — the `.dvu` overlay name encoded its generation in decimal, not base 36. **Fixed.**

**Java**: `IndexFileNames.fileNameFromGeneration` writes
`Long.toString(gen, Character.MAX_RADIX)`, and `parseGeneration` reads it back
the same way. This port's `deletes::liv_file_name` already did.

**We did**: `format!("{segment_name}_{gen}_LuceneRustDVU_{field_number}.dvu")`.
Decimal and base 36 agree only below generation 10; from 10 on,
`_0_10_LuceneRustDVU_3.dvu` reads back as generation **36**.

**Consequence**: bounded but real — `inflate_gens` would push that segment's
three next-write counters past a generation that does not exist. Over-inflation
never collides with a live file, so nothing is lost; what was false was F-15's
own claim that the name is shaped so `parseGeneration` reads the generation
back out of it.

**Fixed**: `lucene_util::base36::to_base36(gen)`, same as `liv_file_name`.
**Test**: `every_generational_file_name_round_trips_through_parse_generation`
asserts the property for `gen in 1..=100` over *both* generational names, which
is what makes it cross the base-36 boundary rather than testing one value below
it. Verified to fail against the decimal version at generation 10.

#### F-28 `[CORRECTNESS]` — a flush between `prepare_commit` and `finish_commit` silently discarded its segment and its resolved deletes. **Fixed.**

**Java**: `finishCommit` updates generation bookkeeping and renames; it does
**not** replace the live `segmentInfos`, so documents added between the two
phases survive into the next commit.

**We did**: `finish_commit` installs the `SegmentInfos` snapshot
`prepare_commit` took and clears `flushed_segments`. Nothing stopped
`add_document` → `maybe_flush` → `flush()` from running in between, and that
`flush()` both appends to `flushed_segments` *and* mutates
`self.segment_infos.segments` (new `del_gen`, new `.liv`) through
`apply_all_deletes_and_updates`. Both were thrown away.

The lost-documents half predates this batch (c3 introduced
`flushed_segments`); the lost-**deletes** half is new, because
`apply_all_deletes_and_updates` is what now writes into the committed list
behind the snapshot's back. The existing `PreparedCommitPending` guard covers
only the operations that write their own `segments_N`, so the buffering entry
points walked straight past it.

**Fixed by deferral, not by refusal**: `maybe_flush` is a no-op while a commit
is prepared, so buffered documents stay in the buffer and buffered deletes stay
in the queue, and both land in the next commit — which is where Java puts them
too (they are not in `pendingCommit`). An *explicit* `flush()` in that window is
refused with `PreparedCommitPending("flush")` rather than silently deferred,
since a caller asking for it deserves to be told. The cost is that the
RAM/document thresholds do not apply inside the prepare → finish window; that
window is a two-phase commit's activation step, by design a short one, and it
is documented at `maybe_flush`.

**Tests**:
`documents_and_deletes_buffered_during_a_prepared_commit_survive_into_the_next_one`
(with `max_buffered_docs = 2`, so the adds used to trip the flush) and
`an_explicit_flush_is_refused_while_a_commit_is_prepared`.

### Advisories acted on

| # | What | Resolution |
|---|---|---|
| A1 | the apply path deep-cloned every packet, then dropped the original | `BufferedUpdatesStream::take_pending` moves instead — one fewer full copy of every buffered term and update value |
| A2 | `open_segment_for_deletes` copied the whole `.tim`/`.tip`/`.tmd`/`.doc` off the mapping every round | holds the directory's `Input` and borrows it, as the deleted FFI code did; on `MmapDirectory` this was heap-copying an entire segment's postings per buffered-delete round |
| A3 | delete-query ordering went through `format!("{:?}", …)` | `DeleteQuery` derives `Ord`; sorted by value, no allocation per comparison and no drift if a `Debug` impl changes |
| A5 | `set_buffered_deletes_gen` silently swallowed a double publish where Java throws | `debug_assert_eq!` so the bug the guard exists to prevent is visible in test builds |
| A6 | the merged-segment/`buffered_deletes_gen == -1` safety argument was true but unenforced | `debug_assert!(!self.updates_stream.any())` at the top of `execute_merge` and `apply_merge` |
| A8 | two methods whose doc comments described call sites that do not exist | both now say "not called by this port", and why |
| A10 | nothing handed real Lucene the one *real-Lucene-format* file this batch newly writes (`.liv`) | `write_block_segment_fixture` now adds a final block and **buffered-deletes it**; `VerifyBlockSegment` asserts `numDocs == maxDoc - 4`, that the deleted term matches nothing through an `IndexSearcher`, and `CheckIndex` cross-checks the bitset against `segments_N`'s `delCount`. Confirmed the segment carries a real `_0_1.liv` — F-3's extra generation, and Lucene accepts it |
| A11 | the seqNo monotonicity test covered six of the mutating methods, not all of them | extended to `update_doc_values`, `update_numeric_doc_value`, `update_binary_doc_value`, `soft_update_document(s)`, `add_document_with_custom_freq_terms` and the `MatchAll` → `deleteAll` path |

### Advisories recorded, not acted on

- **A4 — `sci_id` is not regenerated when a generation advances.** Java's
  `advanceDelGen`/`advanceDocValuesGen`/`setBufferedDeletesGen` all call
  `generationAdvanced()`, which re-randomises the id; its javadoc says the id
  "changes each time the segment changes due to a delete, doc-value or field
  update". Here two commits of the same segment report the same id. Nothing in
  Lucene validates it; what is lost is its use as a change token (segment
  replication). It is a strictly *larger* version of b9's F-13 than the one
  this batch was asked to close, and it wants the same one-pass treatment
  across every generation-advancing site. **Carry-over**, added to the ledger.
- **A7 — the sequence number never crosses the FFI boundary.**
  `ffi_writer_add_document`/`update_document`/`delete_documents` all discard it.
  Surfacing it needs a new `out_seq_no: *mut i64` parameter on three exported
  functions, i.e. an **ABI change**, which is not something to do silently
  inside a sweep batch — especially as the JVM side has not asked for it yet.
  The value is real (OpenSearch's `InternalEngine` uses Lucene's returned seqNo
  directly), so this is recorded as a scope decision rather than an oversight.
  **Carry-over.**
- **A9 — `delete_documents_by_query(MatchAll)` takes its seqNo before
  `delete_all()` can fail.** Java takes it inside `deleteAll()` after the abort
  succeeds. Monotonicity is preserved either way; the only difference is that a
  failed call still consumes a number, which no contract forbids. Cosmetic;
  recorded.

---

## Summary

**28 findings**: 9 CORRECTNESS (8 fixed, 1 recorded with a named blocker),
11 MISSING (10 fixed, 1 declined with reasoning), 4 PERF (1 fixed, 3 recorded
with measurements or reasons), 4 INTENTIONAL. Three of the CORRECTNESS
findings (F-26/F-27/F-28) came from the Tier-2 review, and two more (F-24/F-25)
from this batch's own review pass — five of the eight were defects this batch
introduced, which is the honest shape of a change that makes a previously
immutable structure mutable.

**Fixed, with tests:**

| # | What |
|---|---|
| F-7 | every mutating `IndexWriter` method returns a `long` sequence number, starting at 1 |
| F-8 | `updateDocument`/`deleteDocuments(Term...)` are buffered, with the `docIDUpto` ordering guarantee |
| F-9 | `delGen`/`bufferedDeletesGen` decide which segments a delete may reach |
| F-10 | `addDocuments`/`updateDocuments` block adds, with `hasBlocks` — confirmed by real Lucene |
| F-11 | `deleteDocuments(Query...)` over a closed `DeleteQuery` enum, incl. LUCENE-6379 |
| F-12 | `softUpdateDocument`/`softUpdateDocuments` |
| F-13 | `updateDocValues`/`updateNumericDocValue`/`updateBinaryDocValue`, incl. null-means-`reset` |
| F-16 | `rollback`/`deleteAll` clear the delete queue and the updates stream |
| F-17 | `add_document` does not allocate a one-element `Vec` on the write path's hot loop |
| F-18 | `SegmentCommitInfo`'s four transient generation fields |
| F-19 | `inflateGens`' per-segment half + `IndexFileNames.parseGeneration` |
| F-20 | `sci_id` is written for every freshly flushed or merged segment (b9 F-13) |
| F-21 | `BinaryDocValuesFieldUpdates` (b6 #2) |
| F-22 | the FFI delete path is buffered, matching Java, instead of committing per document |
| F-24 | `rollbackSegments` — a rollback after a buffered delete no longer strands the in-memory view |
| F-25 | a doc-values update no longer reaches a document the same packet just deleted |
| F-26 | a rollback no longer rewinds the delete generation past segments that carry it |
| F-27 | generational file names encode their generation in base 36, as Java's do |
| F-28 | a flush between `prepare_commit` and `finish_commit` no longer discards its work |

**Open, recorded:**

| # | What | Blocker |
|---|---|---|
| F-15 | a segment carrying doc-values updates is not readable by real Lucene | needs a generational `.dvd`/`.dvm` writer (b6's declared scope) |
| F-23 | `norms::parse_meta` does not validate field numbers | declined with reasoning; assign to the batch that lands a shared-`FieldInfos` reader |
| F-3 | one extra `.liv` generation per flush-with-deletes | needs a second in-RAM resolution path to save one small file |
| F-4 | delete terms are a sorted `Vec`, not `PrefixCodedTerms` | 8.3% of a structure that lives for one call stack |
| F-14 | one segment open per apply pass, no reader pool | same milestone as NRT `getReader` |
| A4 | `sci_id` not regenerated when a generation advances | wants one pass across every generation-advancing site |
| A7 | the seqNo does not cross the FFI boundary | needs an ABI change on three exported functions |

**New tests**: 16 in `buffered_updates.rs`, 35 in `index_writer.rs`,
13 in `doc_values_updates.rs`, 4 in `index_file_deleter.rs`, 1 new + 4 rewritten
in `lucene-ffi/writer.rs`, and one new real-Lucene verifier
(`VerifyBlockSegment`).

**The contract tests were checked to *discriminate***, not merely to pass —
each of the three rules was broken in turn and the tests watched to fail:

| broken | tests that failed |
|---|---|
| `FrozenBufferedUpdates::applies_to` forced to `true` (every packet reaches every segment) | `a_delete_applies_to_the_segments_flushed_before_it_and_not_to_the_ones_after`, `a_packet_applies_to_older_segments_and_not_to_newer_ones` |
| `FrozenBufferedUpdates::limit_for` forced to `MAX_DOC_ID_UPTO` (no `docIDUpto` bound) | `a_delete_does_not_reach_a_document_added_after_it_in_the_same_segment`, `interleaved_adds_updates_and_deletes_produce_the_expected_visible_set`, `soft_update_document_adds_the_new_doc_and_marks_the_old_one_without_deleting_it`, `only_a_segments_own_private_packet_honours_the_doc_id_upto_limit` |
| `rollbackSegments` restore removed (F-24) | `a_rollback_after_a_buffered_delete_was_applied_restores_the_committed_segment_list` |
| same-packet delete/update ordering removed (F-25) | `a_doc_values_update_skips_a_document_the_same_packet_just_deleted` |

That is the evidence that the two mechanisms the batch exists to provide are
actually load-bearing in the tests, rather than the tests passing for some
other reason.
