# M4 — Write path hardened for production

> **Goal:** the Rust `IndexWriter` is crash-safe, concurrent, and
> byte-interoperable with Java's — indexes written by either engine fully
> usable by the other.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M3](m3-write-path-proven.md) |
| **Unblocks** | [M5](m5-engine-integration.md) |
| **Status** | done (2026-09-25) -- T4.1 to T4.6 done, every acceptance criterion met |

---

## Why this milestone exists

[M3](m3-write-path-proven.md) proves the *bytes* are right. This milestone
proves the *behaviour* is right: that the writer survives crashes, concurrent
use, and merges without producing an index that is subtly wrong.

`crates/lucene-index/` already has the pieces — `index_writer.rs`,
`indexing_chain.rs`, `segment_writer.rs`, `merge.rs`, `merge_policy.rs`,
`deletes.rs`, `term_delete.rs`, `points_delete.rs`, `update_document.rs`,
`check_index.rs`. `docs/parity.md` describes `IndexWriter` accurately as a
"facade over already-built primitives". What it is not yet is a thing you would
point production traffic at: it is single-threaded, the merge scheduler is a
policy without an executor, and nothing has ever killed it mid-commit to see
what happens.

Everything here is a precondition for [M5](m5-engine-integration.md). An
OpenSearch shard that loses data on restart is worse than no Rust engine at
all.

---

## Scope

### In scope

- The remaining doc-values and DISI write gaps.
- Merge-time reordering of the formats that sorted merges currently skip.
- Concurrent indexing and real merge execution.
- Crash safety, proven by fuzzing rather than by inspection.
- Differential fuzzing of operation streams against Java `IndexWriter`.

### Out of scope

- OpenSearch-level concerns — translog, sequence numbers, soft-delete
  retention. Those are [M5](m5-engine-integration.md).
- Write-path throughput optimisation beyond avoiding obvious pathologies. M6
  handles sustained performance.
- New query or search features.

---

## Scope decision required before starting: vector search

Decide this **first**, because it changes M5's scope materially and cannot be
retrofitted cheaply.

Current state, from `docs/parity.md`:

> **first slice only (task #219): flat storage read+write and exact
> brute-force KNN**

`Lucene99FlatVectorsFormat` (`.vec`/`.vem`) round-trips and brute-force KNN
works. `Lucene99HnswVectorsFormat` (`.vex`) — the actual graph — does not
exist.

- **If OpenSearch k-NN is in the target matrix**, HNSW is a milestone of its
  own, sitting between M4 and M5. It is a graph construction and search
  algorithm with its own on-disk format, its own differential-testing surface,
  and its own performance gate. It is not an increment on flat storage.
- **If it is not**, record k-NN as Java-served in `docs/parity.md`, and make
  sure M5's engine correctly delegates vector fields rather than silently
  dropping them.

Write the decision into this file before starting T4.1.

> **Decision (2026-09-24): k-NN is in the native matrix.** The premise above
> is out of date: `Lucene99HnswVectorsFormat` is ported -- graph build, search
> and merge (`lucene-codecs/src/hnsw.rs`, `hnsw_vectors.rs`), written at flush
> by `IndexWriter::add_document_with_vectors` and rebuilt or reused at merge
> (`merge.rs::merge_vectors`) -- and real Lucene reads it: `VerifyVectors`
> walks every level's neighbours and runs a `TopKnnCollector` search over the
> Rust-built graph, `VerifyVectorSegment` runs `KnnFloatVectorQuery` /
> `KnnByteVectorQuery` against Lucene's own brute force plus `CheckIndex`,
> and `VerifySortedSegment` searches a graph rebuilt by a sorted merge. So no
> separate HNSW milestone is needed, and M5 serves vector fields natively
> rather than delegating them. Scalar-quantized formats
> (`Lucene99ScalarQuantizedVectorsFormat` and later) are not ported and stay
> out of the matrix until a milestone takes them on.

---

## Tasks

### T4.1 — Close the remaining doc-values write gaps

- **Sparse `IndexedDISI.writeBitSet`.** `docs/parity.md` records it as
  "deferred to Phase 5 (write path), needed once sparse norms/doc-values are
  written". Target `crates/lucene-codecs/src/indexed_disi.rs`. The read side
  already decodes SPARSE/DENSE/ALL blocks against real fixtures, and
  `VerifySparseNumericDocValues.java` already exists to check the reverse
  direction — extend it rather than writing a new verifier.
- **The remaining doc-values kind.** `Lucene90DocValuesConsumer`'s write path
  currently covers four of five. Close the fifth and extend
  `VerifyDocValues.java` to cover it.

### T4.2 — Reorder postings and points in sorted merges

`docs/parity.md` and `PLAN.md` both record the current limit: task #205 made
`merge_sorted_stored_only_segments` reorder stored fields, doc values, norms
and term vectors by sort key — but **not postings or points**, which are only
reachable through the concatenation-order merge.

That means an index-sorted segment produced by a merge has its postings in a
different document order than its doc values. Consumers that assume the sort
holds across all formats — which is the entire point of index sorting — get
wrong answers.

Target `crates/lucene-index/src/merge.rs`, reusing the existing `doc_order`
parameter threading rather than building new sort machinery.

This task and T3.5 together determine whether index sorting is a supported
feature or a fenced-off one. Keep the two decisions consistent.

> **Done (2026-09-24).** The premise above is out of date too: `merge_segments`
> already reordered postings, points and vectors through `build_doc_id_maps`.
> What was missing was points *reaching* a merge at all -- `IndexWriter` wrote
> none at flush, so no segment it produced had any. Now
> `IndexWriter::add_points_field` indexes points at flush
> (`Lucene90PointsWriter`, `NumericUtils`' sortable encodings) and
> `execute_merge` hands every source's points to `merge_points`, so they
> survive concatenation and index-sorted merges alike.
> `VerifyPointsSegment.java` checks every point of every document through real
> Lucene in a flushed, a merged and a sorted-merged index. The BKD writer then
> went through port -> benchmark -> optimise: the first faithful port ran at
> 0.76x (flush) / 0.58x (merge) Lucene; one sort per 1-D field
> (`writeField1Dim`), a select per multi-dimensional node
> (`MutablePointTreeReaderUtils.partition`) and a flat point buffer
> (`MutablePointTree`) bring it to 3.68x / 2.08x
> (`scripts/bench-micro.sh --bench points_write`).

### T4.3 — Concurrent indexing and real merge execution

Three separate pieces, often conflated:

- **Multiple DWPTs.** `PLAN.md` §3.5 point 6 sets the design: one owner per
  DWPT, no locking in the indexing hot path, channel-based handoff to flush.
  Follow that rather than porting Java's `synchronized` structure.
- **A merge scheduler.** `merge_policy.rs` implements `TieredMergePolicy`'s
  *decision* — which segments to merge. Nothing *executes* those decisions
  concurrently with indexing. That executor is this task.
- **`IndexFileDeleter`.** Reference-counted file lifecycle. Without it,
  merged-away segments are either deleted while still referenced (corruption)
  or never deleted (unbounded disk growth). This is also what M2's reader-leak
  test exercises from the other side.

> **Done (2026-09-24).** `crates/lucene-index/src/concurrent_writer.rs`,
> `ConcurrentIndexWriter`: one `IndexWriter` shared by any number of
> threads, in `PLAN.md` §3.5 point 6's shape.
>
> - **DWPTs.** One buffer per slot, each behind its own `Mutex`; a thread
>   adding a document locks one free slot. A full buffer's segment is built
>   by the thread that filled it, with no lock held, through
>   `IndexingConfig::build_and_write_segment` (the build half of
>   `IndexWriter`, split out behind an `Arc` for exactly this). Segments
>   publish in ticket order, as `DocumentsWriterFlushQueue` does.
> - **Deletes.** One append-only log of every delete behind its own lock
>   (`DocumentsWriterDeleteQueue`), which also hands out sequence numbers.
>   After each add, still holding its slot, a thread applies the entries the
>   slot has not seen, limited to the documents it held before the add
>   (`DeleteSlice.apply`); an update appends its delete in that same step,
>   after its document (`finishDocuments`). A ticket freezes what the
>   published segments owe and publishes it just before its own segment, so
>   a delete issued while a segment is being built still reaches it. A
>   buffer's own deletes are resolved by its flushing thread before the
>   publish: term deletes against the terms it has just inverted
>   (`FreqProxTermsWriter.applyDeletes`), queries and doc-values updates
>   against the written segment.
> - **Commits.** A commit takes every slot at once, tickets each buffer, then
>   takes a *cut* -- a ticket carrying only the published segments' deletes
>   (Java's global-only `FlushTicket`). With every slot held no update is half
>   done, so each operation is wholly in the commit or wholly out of it.
>   Segments ticketed after the cut build meanwhile but publish only once
>   `segments_N` is written (Java's `blockedFlushes`).
> - **Merges.** `maybe_merge` claims a merge under the control lock, reads and
>   writes with no lock (`IndexingConfig::run_merge`), and takes the lock
>   again to publish it -- in memory only, as `commitMerge` does; the next
>   commit makes it durable. Deletes made meanwhile are carried onto the
>   merged segment through the merge's doc maps (`commitMergedDeletes`). A
>   merge whose source took a doc-values update meanwhile is abandoned, and
>   proposed again on the next round. `run_merges` is `ConcurrentMergeScheduler`'s
>   thread, on a thread the caller owns.
> - **`IndexFileDeleter`** already existed. What is new is that a running
>   merge holds its sources' files (`hold_segment_files`), so a commit cannot
>   reclaim them from under it.
> - **Failures.** A build that fails or panics still retires its ticket
>   (`DocumentsWriterPerThread.abort`): its files are deleted, and its
>   frozen packet is applied. A failed merge deletes what it wrote.
>
> Found on the way: `checkpoint_committed` ran its non-commit checkpoint
> over the committed segments only, so a merge commit deleted the files of
> segments flushed but not yet committed. It now checkpoints over
> `live_infos()`. Single-threaded, merges happen only inside `commit`, after
> the flush, which is why nothing had hit it.
>
> The Tier-2 review of the first version found two blocking defects, both now
> fixed and each pinned by a test that fails with the defect back in:
>
> - An update appended its delete before its document reached a slot, and a
>   commit handed over the whole log. So a commit could make the old version's
>   delete durable without the new version -- the id gone. The fix is the
>   update ordering and the commit cut above.
> - A merge published by writing `segments_N`, making a mid-indexing state
>   durable. Merges now publish in memory only.
>
> The same review drove other fixes: the abandoned ticket's packet left in
> the stream; `keep_fully_deleted_segments` lost with the merge policy;
> sequence numbers that did not follow the log and restarted across the
> hand-over; idle slots applying every delete; and a commit that could wait
> on other threads' later segments indefinitely.
>
> Verified by:
>
> - the module's 13 tests, each ending in `CheckIndex` where it leaves an
>   index:
>   - four threads adding while a merge thread runs;
>   - three threads updating and deleting their own ids while merges run,
>     then the exact last version of every id;
>   - forty commits racing two updating threads, each commit holding every
>     id exactly once;
>   - a delete reaching a segment being built when it was issued, and one
>     reaching the first of two in-flight segments through the second's
>     ticket;
>   - a delete not reaching a document added after it;
>   - a sorted flush resolving its own term, query and doc-values updates
>     through the sort map;
>   - a buffer its own deletes empty never published;
>   - a failed build abandoned, with its packet still applied;
>   - a delete made during a merge carried onto the merged segment, and a
>     doc-values update abandoning one;
>   - the hand-back to the single-threaded writer, sequence numbers included.
> - `scripts/verify-interop.sh`'s eleventh direction: four threads index
>   6 000 documents with a merge thread running and a delete issued
>   mid-merge; real Lucene and this port each read every document back and
>   run their `CheckIndex`.
> - `crates/lucene-search/examples/concurrent_soak.rs`, the endurance run:
>   indexing threads update and delete their own ids at random, a merge thread
>   merges, and every commit is checked **exactly**. `commit()` returns the
>   sequence number of the last operation it holds, as Java's does, and the
>   committed index must equal the model of every operation numbered up to
>   it. `CheckIndex` runs every 25 commits, and resident memory and open file
>   descriptors are reported throughout. CI runs it for two minutes; the soak
>   result is below.
>
> The soak's first run found two leaks, both caches keyed by segment name
> that were never pruned. Single-threaded they grew by a segment per commit
> and went unnoticed. Under this writer they grew by thousands of segments a
> minute, about 5 MB of resident memory a minute.
>
> - The deleter's recorded file sets (`si_files`) now drop a segment once no
>   file of it is referenced.
> - The writer's `segment_versions` cache is pruned to the segments in play
>   whenever a commit is stamped.
>
> After the fix a three-minute run stays flat: RSS 9.6 to 9.9 MB across
> 1.3 M operations and 6 000 merges.
>
> Benchmarked (`scripts/bench-micro.sh --bench concurrent_index --pin 0-3`,
> 4 vCPUs, 100 000 documents of a stored id and a twelve-word body, 10 000 per
> buffer, no merges; median of 3 interleaved reps, noise floor 1.12x):
>
> | case | this port | Lucene 10.5.0 | ratio |
> |---|---|---|---|
> | `add_t1` | 5 529 ns/doc | 7 606 ns/doc | 1.38x |
> | `add_t4` | 2 531 ns/doc | 4 299 ns/doc | 1.70x |
> | `update_t4` | 6 590 ns/doc | 5 351 ns/doc | **0.81x** |
>
> Re-measured after M4 was merged onto `main`'s M3 term-dictionary writer
> (the one pinned byte for byte against Lucene): `add_t1` 1.25x, `add_t4`
> 1.60x, `update_t4` 0.82x, noise floor 1.18x -- the same picture.
>
> Optimising took `add_t4` from 0.70x and `update_t4` from 0.30x. Four
> changes did it:
>
> - The benchmark's counting allocator, one shared atomic per allocation,
>   was a third of the four-thread profile. It now counts only for the memory
>   case.
> - The control lock is off the delete path.
> - The delete log holds its lock only to append or copy pointers, with
>   `Arc`-shared terms.
> - A buffer's own deletes are resolved outside the lock, and in memory.
>
> **`update_t4` is still slower, and why.** Every ticket's frozen packet of
> the published segments' deletes is resolved against every published
> segment *under the control lock*, so publishes serialise, and a thread that
> wants a ticket waits behind one. Java applies those packets outside
> `IndexWriter`'s monitor, per segment (`FrozenBufferedUpdates.apply` over
> `ReadersAndUpdates`, each with its own lock). Porting that means per-segment
> state this port does not have: a reader pool, and a segment list readable
> without the control lock. That is the next step for this path. The
> add-only cases, which are what bulk indexing is, are already ahead.

### T4.4 — Crash fuzzing

The acceptance criterion the milestone lives or dies by.

- A harness that drives a randomised operation stream — add, update, delete,
  commit, merge — against a real on-disk directory.
- `kill -9` at randomised points, including *inside* a commit, *inside* a
  merge, and between `prepare_commit` and `finish_commit` (the two-phase
  commit path already exists as `ffi_writer_prepare_commit` /
  `ffi_writer_finish_commit`).
- On restart, assert: the index opens, real Lucene's `CheckIndex` passes, and
  the visible state is **exactly** the last durable commit — no partial
  commit, no torn segment, no resurrection of deleted documents.
- Run under both `FsDirectory` and `MmapDirectory`.

`PLAN.md` §4 names two-phase commit and crash recovery as risk #2 for Phase 6.
Fuzzing it here, before OpenSearch is involved, is what keeps it from becoming
an M5 problem.

> **Done (2026-09-25).**
> `crates/lucene-search/examples/crash_fuzz.rs` runs a seeded stream of adds,
> updates and deletes by term, flushes (explicit and automatic), commits and
> two-phase commits (`prepare_commit` then `finish_commit`), with merges firing
> inside commits. It crashes the stream in one of two ways:
>
> - **Power loss** (the risk section's fault-injection layer).
>   `lucene_store::crashing_directory::CrashingDirectory`, the `crash()` half
>   of Lucene's `MockDirectoryWrapper`, fails a directory operation drawn
>   uniformly over the whole run and everything after it. It then leaves
>   unsynced files kept, truncated, zeroed or gone, and keeps only a prefix of
>   the creates, renames and deletes not yet published by `syncMetaData`.
> - **`kill -9`** of a child process at a random moment.
>
> After the crash the index must open and hold exactly the last durable
> commit, or the commit in flight when it crashed, compared id by id and
> version by version. It must also pass this port's `CheckIndex` and, with
> `--java-cp`, real Lucene's. Finally a new writer must recover: add, commit
> and check again. The restarted side alternates `FsDirectory` and
> `MmapDirectory` by seed. `scripts/crash-fuzz.sh` runs 150 power-loss seeds,
> 25 more with Lucene's `CheckIndex`, and 40 `kill -9` seeds; it is part of
> CI's `write-path` job, together with 40 concurrent seeds (below).
>
> Measured: with `pending_segments_N`'s fsync removed, seed 0 fails
> immediately (a commit whose `segments_N` was renamed but never synced loses
> the whole index).
>
> The harness also fails a round when:
> - a crash-free run errors, or any writer error is not the crash itself;
> - the recovered directory holds any file its commit does not reference
>   (`IndexFileDeleter` must reclaim the crash's leftovers);
> - `create_output` is asked for a name that exists (`CREATE_NEW`).
>
> A fixed run of 100 or more seeds must also have produced:
> - a crash inside an automatic flush;
> - a crash inside a two-phase commit;
> - a crash at the publish rename;
> - both outcomes.
>
> Most `kill -9` rounds must land mid-run. A seed replays its op stream and
> crash point exactly; segment ids are random, so the bytes differ between
> replays, and a kill's timing is not reproducible.
>
> **Blind spots:**
> - Only an ordered-metadata filesystem is modelled, and a file is only ever
>   damaged whole, never torn inside a synced region.
> - The oracle accepts the commit in flight **only** when the crash came
>   after its `segments_N` rename (`CrashingDirectory::published`). Such a
>   rename may or may not have been made durable by the power loss. A crash
>   anywhere before it must leave the previous commit, exactly. `--kill`
>   cannot see the rename and accepts a commit that journaled its start.
> - Crash points are drawn over directory *operations*. A write into an
>   already-open output cannot fail on its own; only the next directory call
>   can.

> **Crash campaign: breadth instead of 24 hours.** The criterion first asked
> for 24 hours of fuzzing. A serial run of one process mostly repeats one
> condition, so the time went into varied conditions run in parallel.
>
> - **Varied conditions.** Each seed draws its own operation mix (balanced,
>   update-, delete-, commit- or flush-heavy), a buffer of 2 to 16 documents,
>   and its own merge policy.
> - **`--concurrent`.** A power-loss mode for the `ConcurrentIndexWriter`
>   (T4.3): 2 to 4 indexing threads, its merge thread and a committer, cut
>   at a random directory operation on whichever thread is there. What
>   survives must be a clean prefix of the operations by sequence number,
>   reaching at least the last commit that returned. It catches both
>   concurrency defects the T4.3 review found: with the commit-cut bug put
>   back it fails on seed 0, and with merges writing their own `segments_N`
>   on seed 1.
> - **`scripts/crash-storm.sh`.** One worker per core, each on its own
>   seeds, cycling power loss, concurrent power loss and `kill -9` over
>   streams of 120 to 1 500 operations. Real Lucene's `CheckIndex` runs on
>   one batch in five. With `--load`, CPU burners and an fsync-heavy disk
>   writer run alongside, so thread interleavings and sync latencies are
>   not the quiet machine's. It stops at the first failure with the command
>   that replays it. It was seen failing with the commit-cut bug put back.
>
> Runs, all passing:
>
> | run | length | rounds | detail |
> |---|---|---|---|
> | serial power-loss soak, real Lucene's `CheckIndex` on every round | 7 h 03 m | 7 373 seeds | stopped for the campaign |
> | `crash-storm.sh --load`, 4 workers | 1 h | 1 510 | 760 power-loss, 510 concurrent power-loss, 240 `kill -9`; 654 recovered to the in-flight commit, 856 to the last commit; about 300 with real Lucene's `CheckIndex` |
> | `concurrent_soak`, 2 threads, a merge thread and a committer | 3 h 50 m | 97 117 commits | every commit checked exactly, 87.8 M operations |
>
> **Resources.** The serial soak held its RSS at 5.9 to 6.0 MB and 3 to 5
> file descriptors from start to end. The concurrent soak held 8.4 to 8.8 MB
> and 4 to 8 descriptors across its 3 h 50 m (sampled every 10 min, after
> the two cache leaks it found were fixed -- see T4.3).

### T4.5 — Differential operation-stream fuzzing against Java

The strongest available correctness check, and the natural extension of the
project's differential-testing backbone into the write path.

- Generate a randomised, seeded operation stream.
- Apply it to both this port's `IndexWriter` and a real Java `IndexWriter`.
- Compare the resulting indexes: same live document count, same term
  dictionary contents, same doc-values, same query results for a generated
  query set.
- Exact byte-identity is **not** the criterion — merge timing and block
  splitting legitimately differ. Semantic equivalence is.
- Seeds must be recorded and replayable; a fuzz failure nobody can reproduce
  is not a finding.

> **Done (2026-09-24).** `scripts/op-stream-fuzz.sh` (CI's `write-path` job)
> runs 1000 seeded streams of 200 operations through real Lucene 10.5.0's
> `IndexWriter` (`fixtures/src/OpStreamFuzz.java`) and this port's
> (`crates/lucene-search/examples/op_stream_fuzz.rs`).
>
> The two halves share one generator, and each stream mixes:
> - adds and `updateDocument`;
> - deletes by id and by a body word (which reach buffered documents too);
> - `updateNumericDocValue`;
> - one pick in five targeting any id ever issued, so an update is sometimes
>   a plain add, and a delete or doc-values update sometimes matches nothing;
> - flushes (explicit, and automatic at 2-21 buffered docs) and commits;
> - merges, on each engine's own schedule.
>
> Documents are sparse in one SORTED field.
>
> Each index is dumped semantically and the dumps must match line for line:
> - every live document with its version, NUMERIC and SORTED doc values and
>   points;
> - the live documents each of 23 body terms matches (a scored `TermQuery`
>   on both sides);
> - the live documents each of 50 phrases matches, and one point range (each
>   engine's own `PointValues.intersect` walk).
>
> Segment layout, merge timing and scores are deliberately left out. All 1000
> seeds agree, 46 596 live documents compared. The two engines' histories
> genuinely differ: they end with 3 840 (Lucene) and 3 674 (this port)
> segments over the 1000 indexes.
>
> A seed's dumps are kept on failure, and `--seeds S..S+1` replays it.
> Measured: an off-by-one in which buffered documents a delete applies to
> (`doc <= limit`) fails the first seed, losing every updated document.
>
> **Blind spots:**
> - Stored fields: the Rust reader returns doc ids, not stored documents.
> - Scores: BM25 statistics include deleted documents until a merge drops
>   them, so they legitimately depend on merge timing.
> - Operations this port's writer does not have yet: `forceMerge`,
>   `addIndexes`, soft deletes, binary doc-values updates in the stream.

### T4.6 — Bidirectional interoperability matrix

The explicit statement of what "interoperable" means, tested rather than
asserted:

- Java writes → Rust reads (already covered by the `Gen*.java` fixtures)
- Rust writes → Java reads (M3's `Verify*.java`)
- Java writes → Rust **appends** → Java reads
- Rust writes → Java **appends** → Rust reads
- Java writes → Rust **merges** → Java reads

The last three are new and are what an incremental OpenSearch adoption
actually does — a shard will have segments from both engines simultaneously.

> **Done (2026-09-24).** `scripts/verify-interop.sh` (in CI's `write-path`
> job) drives all five directions plus three more -- a Rust merge of mixed
> Rust and Java segments, and a delete by term in each direction -- over one
> index, with `fixtures/src/InteropIndex.java` and
> `crates/lucene-search/examples/interop.rs` sharing one document table. It
> found three defects on its first run, none visible to any single-engine
> test: this port could not read a **compound** segment anywhere in its
> write path (and Java flushes compound segments by default), so its merge
> would have silently dropped a Java segment's postings, doc values and
> points, its delete would have silently matched nothing, and its own
> `CheckIndex` failed every Java segment. `CompoundReader`
> (`Lucene90CompoundReader` as a `Directory`) fixes all three.

---

## Acceptance criteria

- [x] The k-NN scope decision is recorded in this file before T4.1 starts.
- [x] A random-op and random-crash fuzz leaves an index that real
      Lucene's `CheckIndex` passes — **every time**, across every seed.
      Originally a 24-hour run. Met instead by the crash campaign (T4.4):
      7 h serial with Lucene's `CheckIndex` on every round, 1 h in parallel
      under load across every crash model including the concurrent writer,
      and 3 h 50 m of exactly checked concurrent commits.
- [x] After every simulated crash, visible state is exactly the last durable
      commit: no partial commits, no resurrected deletions.
- [x] Differential operation-stream fuzzing against Java `IndexWriter` shows
      semantic equivalence across ≥1000 seeds.
- [x] All five directions of the T4.6 interoperability matrix pass.
- [x] Concurrent indexing from multiple threads with merges running produces a
      `CheckIndex`-clean index.
- [x] **No file-handle or memory growth** over the soaks: RSS and fds flat
      over the 7 h crash soak and the 3 h 50 m concurrent soak, after fixing
      the two leaks the latter found (T4.3).
- [x] Index-sorted merges preserve sort order across *every* format, or index
      sorting is explicitly unsupported and refused.
- [x] Per-file line coverage stays ≥95% across every file touched: no file
      below it at the close (workspace total 97.98%).

---

## Risks and unknowns

- **Concurrency bugs do not reproduce.** These failures are timing-dependent,
  and waiting does not make them more likely -- varying the timing does. So
  the campaign runs under CPU and fsync load, with per-seed thread counts
  and stream shapes. Every round prints its seed, thread count and crash
  point, and replays with `crash_fuzz --seed S [--concurrent]`.
- **`kill -9` fidelity.** A process kill does not reproduce every real failure
  mode — it leaves the page cache intact, so it tests process crashes but not
  power loss. Consider a filesystem fault-injection layer for the durability
  claims that actually depend on `fsync` ordering.
- **The differential fuzzer may find divergence that is legitimate.** Merge
  timing, segment counts and file layouts will differ from Java's. The
  comparison must be written at the semantic level from the start, or it will
  drown in false positives and get switched off.
- **Scope creep from T4.3.** Concurrent indexing invites redesigning the whole
  indexing chain. `PLAN.md` §3.5 point 6 already specifies the design; follow
  it, and resist widening.

---

## Exit artifacts

- A crash-fuzzing harness with recorded, replayable seeds
- A differential operation-stream fuzzer
- The T4.6 interoperability matrix as an automated test suite
- `IndexFileDeleter` equivalent and a concurrent merge executor
- An updated `docs/parity.md` covering sparse DISI, the fifth doc-values kind,
  and sorted-merge coverage
- The recorded k-NN scope decision
