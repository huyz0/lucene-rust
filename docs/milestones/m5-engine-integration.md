# M5 — OpenSearch indexing from Rust

> **Goal:** an OpenSearch shard fully served by Rust — both indexing and
> search.

| | |
|---|---|
| **Effort** | XL — the largest milestone, and the most dependent on OpenSearch internals |
| **Depends on** | [M2](m2-opensearch-read-path.md) **and** [M4](m4-write-path-hardened.md) |
| **Unblocks** | [M6](m6-production-candidate.md) |
| **Status** | not started |

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

- [ ] OpenSearch `:server` engine tests pass on the Rust engine for the
      supported matrix.
- [ ] The full REST test suite for search, index, get and delete passes.
- [ ] Segment replication works end-to-end between a Rust primary and its
      replicas.
- [ ] Peer recovery brings a new replica to a consistent state, verified by
      comparing document counts and a query result set against the primary.
- [ ] A **mixed cluster** — Rust primary with Java replicas, and Java primary
      with Rust replicas — operates correctly.
- [ ] Sequence numbers, local checkpoint and commit user data are
      byte-compatible with `InternalEngine`'s, verified by having Java code
      paths recover from a Rust-written commit.
- [ ] Optimistic concurrency control rejects conflicting writes identically to
      the Java engine.
- [ ] A Rust-engine panic **fails exactly one shard**; the node survives and
      continues serving other shards.
- [ ] Circuit-breaker accounting reflects Rust-side memory; a deliberate
      memory-pressure test trips the breaker rather than the OOM killer.
- [ ] Aggregation results are identical to the Java engine's across a
      representative aggregation set.

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
