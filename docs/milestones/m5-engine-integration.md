# M5 — OpenSearch indexing from Rust

> **Goal:** an OpenSearch shard fully served by Rust — both indexing and
> search.

| | |
|---|---|
| **Effort** | XL — the largest milestone, and the most dependent on OpenSearch internals |
| **Depends on** | [M2](m2-opensearch-read-path.md) **and** [M4](m4-write-path-hardened.md) — both delivered 2026-09-25 |
| **Unblocks** | [M6](m6-production-candidate.md) |
| **Status** | delivered (2026-09-25): document and segment replication |

---

## Outcome

An OpenSearch 3.8.0 shard can be fully served by Rust: `RustEngine` indexes
through the Rust `IndexWriter`, and searches go through M2's native query
phase. How it is built, its settings, what it refuses and where it behaves
differently are in [`../opensearch-engine.md`](../opensearch-engine.md).

The engine is `InternalEngine` itself, derived from the pinned sources with
only the writer swapped (`opensearch-plugin/tools/derive_engine.py`). This is
the answer to this file's biggest risk: sequence numbers, checkpoints, commit
user data, the version map and history retention are OpenSearch's own code
rather than a re-implementation of it.

Two things differ from the plan below:

- **Segment replication needs one hook into OpenSearch.**
  `CopyState` reads a segment-replication primary's checkpoint through
  `EngineBackedIndexer.lastRefreshedCheckpoint()`. That call answers only for
  an `InternalEngine`, and `InternalEngine.lastRefreshedCheckpoint()` is
  `final`, so on its own no plugin engine can be a segment-replication
  primary on 3.8. `RustIndexerFactory` replaces the index's indexer
  factory in `IndexModule` (`onIndexModule`, before any shard exists) with
  one that wraps `RustEngine` in an `EngineBackedIndexer` answering from the
  engine's own checkpoint listener. That is the plugin's one reflective
  write, one per index; `docs/opensearch-engine.md` explains the choice. A
  `RustEngine` built any other way refuses to be a segment-replication
  primary, and an engine test has seen that refusal fire.
  Remote-backed storage stays refused. Both replication modes are verified
  in the cluster proof.
- **Get-by-id and aggregations need no FFI path.** The engine's readers are
  Java `StandardDirectoryReader`s over the Rust writer's commits, so realtime
  get, `_source`, versioning and the aggregation framework run unchanged on
  Rust-written segments. The verification below checks their results against
  OpenSearch's engine, value for value.

The work also fixed Rust write-path defects that only an engine workload
exposed:

- doc-values updates resolved by field number rather than name, which
  corrupted segments Java wrote or a merge renumbered
- no doc-values skip index writer: every `@timestamp` document was refused
- a SORTED_SET shape Lucene's reader could not open
- merges that kept dictionary terms only deleted documents used
- merges that refused a field with no terms
- no `IndexWriter.MAX_DOCS` limit at all

Each fix has a regression test that fails without it.

---

## Why this milestone exists

M2 put Rust behind reads. M4 made the Rust writer trustworthy on its own
terms. This milestone joins them: an `Engine` implementation where the
`IndexWriter` lives in Rust and OpenSearch's durability, replication and
recovery machinery still works.

This is where the port stops being a Lucene problem and becomes an OpenSearch
problem. The remaining work is mostly about matching `InternalEngine`'s
contracts exactly — sequence numbers, checkpoints, commit user data — rather
than about index formats.

`PLAN.md` §4 names the hardest part as risk #2: two-phase commit and translog
recovery semantics. [M4](m4-write-path-hardened.md)'s crash fuzzing is the
mitigation that should already be in place before this milestone starts.

---

## Scope

### In scope

- Soft-deletes write side.
- An `Engine` implementation backed by the Rust `IndexWriter`.
- Segment replication.
- Metadata field parity and the get-by-id fast path.
- Operational integration: circuit breakers, stats, logging, shutdown, failure
  containment.
- Aggregation feeding over batched doc-value cursors.

### Out of scope

- Document replication mode. Segment replication ships first, per `PLAN.md`
  Phase 6 item 3; document replication is a follow-on, not a prerequisite.
- Native aggregation implementations. The framework stays on the JVM; this
  milestone only feeds it efficiently.
- Cross-cluster replication and snapshot/restore, beyond not breaking them.
- Backward-codecs. Out of scope project-wide.

---

## Tasks

### T5.1 — Soft-deletes write side

`docs/parity.md` records the read side as done and is explicit about the gap:

> **ported (task #48), read-side (visibility) only**

`SoftDeletesDirectoryReaderWrapper` / `PendingSoftDeletes` visibility checks
work; `IndexWriter.softUpdateDocument` does not exist. OpenSearch requires
soft-deletes for peer recovery and retention leases — `PLAN.md` §0 calls this
out as a v1 requirement rather than a nice-to-have.

- Implement `softUpdateDocument`: add the new document and mark the old one
  soft-deleted via the configured field, atomically.
- Implement the retention merge policy equivalent
  (`SoftDeletesRetentionMergePolicy`), so soft-deleted documents survive
  merges until the retention lease releases them. Getting this wrong breaks
  replication in a way that only shows up under recovery.
- Extend M4's differential fuzzer to cover soft-delete operation streams.

### T5.2 — The `Engine` implementation

The core of the milestone. Match `InternalEngine`'s observable contracts
exactly:

- **Sequence numbers and local checkpoint.** Every operation gets a seq-no;
  the local checkpoint advances only over a contiguous prefix. Replication
  correctness depends on this precisely, not approximately.
- **Commit user data.** OpenSearch stores recovery state in the segment
  commit's user data map. Byte-level parity here is what makes a Rust-written
  commit recoverable by Java code paths.
- **Refresh → NRT reader.** Map OpenSearch's refresh onto the existing
  `open_if_changed` path from `crates/lucene-search/src/directory_reader.rs`.
- **Flush → commit.** Map onto `prepare_commit`/`finish_commit`, already
  exposed over FFI.
- **The translog stays Java-side.** `PLAN.md` is explicit. The Rust engine
  must expose enough state for the Java translog to drive recovery; it must
  not own the translog.

### T5.3 — Segment replication first

`PLAN.md` Phase 6 item 3 recommends this and the reasoning holds: in segment
replication only primaries index, and replicas reuse M2's read path
unchanged. It is strictly less surface area than document replication.

- Primary indexes through the Rust engine, produces segments.
- Replicas receive and open them through M2's reader.
- Peer recovery works: a new replica catches up from a primary.
- Verify a **mixed** cluster — Rust primary with Java replicas and vice
  versa — since that is what an incremental rollout looks like.

### T5.4 — Metadata field parity and get-by-id

`_source`, `_id`, `_seq_no`, `_primary_term`, `_version` must behave exactly as
the Java engine produces them, because other OpenSearch subsystems read them
directly.

- `_source` retrieval through the stored-fields read path, returned as
  borrowed slices per `PLAN.md` §3.5 point 3 — no intermediate `Vec` churn.
- A get-by-id fast path: term lookup on `_id` over FFI, avoiding a full query
  execution for what is a point lookup.
- Versioning semantics: optimistic concurrency control via `_seq_no` and
  `_primary_term` must reject conflicting writes identically to Java.

### T5.5 — Operational integration

The difference between a demo and something an operator will run:

- **Circuit breakers.** The Rust side reports its RAM usage; OpenSearch's
  breakers account for it. Without this, the JVM's memory accounting is blind
  to the largest allocation in the process.
- **Stats APIs.** Segment counts, memory, merge stats, refresh stats —
  populated from the Rust side so `_cat` and `_stats` are not silently wrong.
- **Slow log hooks.**
- **Graceful shutdown.** Every handle released, every file closed, no
  temp-file litter.
- **Panic → shard-failed, not node-down.** `PLAN.md` §4 risk #3. M2 proved
  panics do not unwind into the JVM; this task proves the resulting error
  fails exactly one shard and the node keeps serving the others.

### T5.6 — Aggregation feeding

Keep OpenSearch's aggregation framework on the JVM and feed it from Rust:

- Batch columnar doc-value reads into shared buffers rather than per-document
  crossings — `PLAN.md` Phase 6 item 5, and the same batching discipline the
  FFI budget in M1 demands.
- Correctness first: aggregation results must be identical to the Java engine's
  before any performance claim is made.

---

## Acceptance criteria

- [x] OpenSearch `:server` engine tests pass on the Rust engine for the
      supported matrix. `InternalEngineTests` 3.8.0, derived to build
      `RustEngine` (`scripts/opensearch-engine-tests.sh`): 117 pass, 0 fail.
      The 34 skipped each carry a reason in
      `opensearch-plugin/tools/derive_engine_tests.py`: refused configurations,
      failure injection through a Java `IndexWriter` or in-memory `Directory`,
      and the documented differences.
- [x] The full REST test suite for search, index, get and delete passes.
      `scripts/verify-opensearch.sh --engine --yaml` runs
      OpenSearch's own REST YAML suites (search, highlight, inner_hits,
      msearch, scroll, count, explain, suggest, get, index, delete, bulk,
      update, mget, exists: 501 tests) on a node where **every** index, system
      indices included, uses the Rust engine (857 shards). The failure set is
      **identical** to a stock node's: the same 4 `_source`-filtering
      warning-header tests fail on both. The first run failed 29 more. Its
      diff found the missing doc-values skip index (OpenSearch gives every
      `@timestamp` one), now written. The rest were completion and
      term-vector mappings, which the node default now sends to OpenSearch's
      engine.
- [x] Segment replication works end-to-end between a Rust primary and its
      replicas (`scripts/verify-opensearch-cluster.sh` §8, via
      `RustIndexerFactory` -- see Outcome):
      - replicas copy the Rust writer's segments;
      - when the primary's node stops, a replica is promoted into the Rust
        engine on the segments it copied;
      - the stopped node rejoins as a replica;
      - the primary relocates to the other Rust node;
      - a replica on the Java node joins;
      - a force merge replicates, and every copy then holds the one merged
        segment;
      - the stats show every primary was built by `RustIndexerFactory`.
      Every copy is checked against the acknowledged writes.
- [x] Peer recovery brings a new replica to a consistent state, verified by
      comparing document counts and a query result set against the primary
      (`scripts/verify-opensearch-cluster.sh`: every copy checked against a
      model of what was acknowledged).
- [x] A **mixed cluster** operates correctly. Rust-engine nodes alongside a
      node with `lucene_rust.engine.node_enabled: false`, primaries relocated
      between the two kinds in both directions, and failover.
- [x] Sequence numbers, local checkpoint and commit user data are
      byte-compatible with `InternalEngine`'s. Java's engine recovers shards
      from Rust-written commits: relocation to the Java node, and
      `EngineWriterDiffTest` against Lucene's `IndexWriter`.
- [x] Optimistic concurrency control rejects conflicting writes identically to
      the Java engine: `if_seq_no`/`if_primary_term` and external versions,
      current and stale, compared item for item (`verify_engine.py`).
- [x] A Rust-engine panic **fails exactly one shard**; the node survives and
      continues serving other shards. The shard recovers from its translog.
- [x] Circuit-breaker accounting reflects Rust-side memory: a lowered
      `lucene_rust_writer` limit trips with a 429, node stats list the
      breaker and its trip, and a refresh releases the accounting.
- [x] Aggregation results are identical to the Java engine's across terms,
      histogram, date_histogram, avg/sum/min/max, cardinality, percentiles,
      nested and top_hits, including after a restart and after SIGKILL.

---

## Risks and unknowns

- **Sequence-number and checkpoint semantics are subtle and unforgiving.** The
  failure mode is not a crash but silent replica divergence, discovered later.
  Test with deliberate, adversarial reordering and gaps, not just the happy
  path.
- **`Engine` SPI surface area.** OpenSearch's `Engine` is large and evolves.
  Pin the target OpenSearch version explicitly, the way Lucene 10.5.0 is
  pinned, and record it in `docs/parity.md`.
- **Retention merge policy correctness.** Getting soft-delete retention wrong
  breaks recovery in ways that appear only under specific replication timings.
  This deserves its own differential test against Java, not just unit tests.
- **The translog boundary.** The Rust engine must expose precisely the state
  the Java translog needs, and no more. An under-specified boundary here shows
  up as unrecoverable shards.
- **Effort concentration.** This is the XL milestone. If it needs to be split,
  the natural seam is segment replication (T5.1–T5.3) as one deliverable and
  operational integration (T5.4–T5.6) as a second.

---

## Exit artifacts

- A Rust-backed `Engine` implementation in `opensearch-plugin/`
- Soft-deletes write side and retention merge policy in `crates/lucene-index/`
- Mixed-cluster replication test suite
- Circuit-breaker and stats bridging
- An updated `docs/parity.md` covering the soft-delete write side and the
  pinned OpenSearch version
