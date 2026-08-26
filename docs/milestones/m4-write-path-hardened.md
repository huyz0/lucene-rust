# M4 — Write path hardened for production

> **Goal:** the Rust `IndexWriter` is crash-safe, concurrent, and
> byte-interoperable with Java's — indexes written by either engine fully
> usable by the other.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M3](m3-write-path-proven.md) |
| **Unblocks** | [M5](m5-engine-integration.md) |
| **Status** | not started |

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

---

## Acceptance criteria

- [ ] The k-NN scope decision is recorded in this file before T4.1 starts.
- [ ] A **24-hour** random-op and random-crash fuzz leaves an index that real
      Lucene's `CheckIndex` passes — **every time**, across every seed.
- [ ] After every simulated crash, visible state is exactly the last durable
      commit: no partial commits, no resurrected deletions.
- [ ] Differential operation-stream fuzzing against Java `IndexWriter` shows
      semantic equivalence across ≥1000 seeds.
- [ ] All five directions of the T4.6 interoperability matrix pass.
- [ ] Concurrent indexing from multiple threads with merges running produces a
      `CheckIndex`-clean index.
- [ ] **No file-handle or memory growth** over the 24-hour soak.
- [ ] Index-sorted merges preserve sort order across *every* format, or index
      sorting is explicitly unsupported and refused.
- [ ] Per-file line coverage stays ≥95% across every file touched.

---

## Risks and unknowns

- **Concurrency bugs do not reproduce.** The 24-hour criterion is deliberately
  a duration rather than an iteration count, because these failures are
  timing-dependent. Record seeds, thread counts and timings for every run.
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
