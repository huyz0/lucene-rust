# Delivery roadmap

End-to-end milestones from today's state to a production-candidate Rust engine inside
OpenSearch. Complements [`PLAN.md`](../PLAN.md) — which is organised by *subsystem phase* —
by organising the same work by *shippable increment*. Each milestone has one goal, a task
list, and acceptance criteria that can be objectively checked.

`PLAN.md` stays the architectural source of truth. `docs/parity.md` stays the per-file
status ledger. This file answers only: **what ships next, and how do we know it's done.**

This file is the summary. Each milestone has a full work order — scope boundaries, task
breakdown with file paths, risks, and exit artifacts — in
[`docs/milestones/`](milestones/README.md).

---

## Where we are (2026-08-27)

| Area | State |
|---|---|
| Tests | 2524 passing, 0 failing |
| Read path (P1–P2) | Effectively complete for the pinned Lucene 10.5.0 codec |
| Search (P3) | Broad: boolean/phrase/dismax/span/wildcard/fuzzy/regexp/points/DV-range, sort, facets, collapse, highlight, explain, concurrent search |
| FFI (P4, Rust half) | 76 `extern "C"` entry points, handle registry, `catch_unwind` on every boundary |
| FFI (P4, Java half) | **Does not exist** — `opensearch-plugin/` is a 2-line README |
| Write path (P5) | Substantial but unevenly proven — see M3 |
| Engine integration (P6) | Not started |
| Performance (P7) | Not started; no Rust-vs-Java comparison has ever been run |
| CI | **Added in M0** — `.github/workflows/ci.yml`: gate on x64 + arm64, plus fixture and write-path jobs |

### Three facts that set the ordering

1. **HEAD does not pass its own pre-commit gate.** `cargo clippy --workspace
   --all-targets -- -D warnings` fails on 11 warnings introduced when the toolchain moved
   1.97.0 → 1.97.1 on 2026-07-14. There is no `rust-toolchain.toml` and no CI, so nothing
   caught it. Everything else is blocked behind a committable tree.

2. **The term dictionary write path has never been read by real Lucene.** Twelve
   `Verify*.java` programs prove Java can read Rust-written stored fields, doc values,
   points, norms, term vectors, FSTs, live docs, compound files and segment infos.
   **None covers `.doc`/`.tim`/`.tip`/`.tmd`.** The postings/blocktree writer's correctness
   rests entirely on round-tripping through *this port's own reader*, which cannot detect a
   shared misreading of the spec. This is the largest correctness risk in the project.

3. **The go/no-go gate in `PLAN.md` has never been evaluated**, and it is far cheaper to
   evaluate than the plan assumes. `PLAN.md` places the decision after the OpenSearch
   read-path integration, but the thing being measured — query CPU on identical segments —
   needs no OpenSearch and no JNI. The read path is already complete, so the benchmark can
   run *now*, against Java Lucene directly. That makes it M1, not a milestone that arrives
   after months of Java plugin work.

---

## Milestone map

```
                         ┌──────────────── pass ────────────────┐
                         │                                      │
M0 ──────▶ M1 (GATE) ────┤          ┌──▶ M2  OpenSearch read ───┼──▶ M5 ──▶ M6
CI green   benchmark     │          │                           │   engine   prod
                         └──────────┴──▶ M3  write proof ──▶ M4 ┘
                         │                                 harden
                         └─ fail ─▶ stop, or ship as a standalone library
```

M2 and M3 are **independent tracks** — one is Java-writes/Rust-reads through OpenSearch, the
other is Rust-writes/Java-reads at the format level. Neither blocks the other, so they can
be worked in parallel or in either order once M1 passes.

**The critical claim of this ordering:** M0 → M1 is short and cheap, and it decides whether
M2 → M6 — the overwhelming majority of remaining effort — should be funded at all.

---

## M0 — Green tree, real CI  ·  delivered

> Full detail, task breakdown and risks: [`docs/milestones/m0-ci-and-green-tree.md`](milestones/m0-ci-and-green-tree.md)

**Goal:** every gate that exists today runs automatically on every change, on the platforms
we claim to support — and HEAD passes them.

**Why first:** the project has a strong local gate (fmt + clippy-deny + ≥95% line coverage)
that nothing enforces. A tree failing its own hook is a tree nobody can commit to. CI is
also the only way the fixture story stays honest: `fixtures/data/` is checked in, and
nothing currently proves it still matches what Lucene 10.5.0 generates.

**Tasks**

- Clear the 11 clippy warnings (`byte_char_slices` ×6, `needless_range_loop` ×5).
- Add `rust-toolchain.toml` pinning the exact toolchain, so a compiler bump becomes a
  deliberate, reviewable change rather than a silent breakage.
- GitHub Actions: `fmt --check`, `clippy -D warnings`, `test --workspace`,
  `llvm-cov --fail-under-lines 95` — on Linux x64 **and** arm64.
- JVM job: regenerate every fixture from Lucene 10.5.0 and assert
  `git diff --exit-code fixtures/data` — proving the checked-in fixtures are genuinely
  Java-produced and have not drifted or been hand-edited.
- JVM job: run all twelve `Verify*.java` reverse verifiers against freshly Rust-written
  output.
- Script the fixture and verifier invocations that currently live as shell snippets in
  `fixtures/README.md`, so CI and humans run an identical command.

**Acceptance criteria**

- A pull request shows all jobs green; a deliberately introduced clippy warning, coverage
  drop, or fixture edit each turn CI red.
- Fixture regeneration is byte-identical to what is committed.
- All twelve reverse verifiers pass unattended in CI, on both architectures.

**Effort:** S — days.

---

## M1 — The performance gate (go/no-go)

> Full detail, task breakdown and risks: [`docs/milestones/m1-performance-gate.md`](milestones/m1-performance-gate.md)

**Goal:** answer, with data, the question `PLAN.md` says must be answered before paying for
the rest of the project: *is Rust search decisively faster than Java Lucene?*

**Why here:** this is the highest-value-per-hour milestone in the plan. The read path is
complete, so the measurement needs no OpenSearch, no JNI, and no new Lucene-format work —
only a harness. Running it second means the go/no-go arrives for the price of a benchmark
instead of the price of a Java plugin. If the answer is no, everything after M1 changes.

**Tasks**

- Build a repeatable harness: identical index, identical query set, lucene-rust vs Java
  Lucene 10.5.0, reporting throughput and p50/p99 latency and CPU — checked in and runnable
  by anyone, not a one-off script.
- Cover the real workload mix: term, boolean, phrase, range, sort-by-field, and facets.
  Use segments written by Java Lucene, so both engines read byte-identical input.
- Profile and close gaps: flamegraphs against Java async-profiler on identical workloads.
  Vectorise the kernels the profile actually implicates — PFOR decode, BKD compare loops,
  bitset ops — rather than the ones assumed in advance.
- Microbenchmark the FFI boundary against the `PLAN.md` budget of <1µs per search call.
  This is a leading indicator only; the real end-to-end overhead is confirmed in M2.
- Revisit the deferred `Weight`/`Scorer`/`BulkScorer` hierarchy **only if** the profile
  shows the current directly-wired scoring blocks dynamic pruning on real queries. It is a
  significant architectural change — do it for a measured reason or not at all.

**Acceptance criteria**

- ≥1.5× throughput versus Java Lucene on ≥80% of the query mix at identical recall.
- **No workload slower than Java.** A regression anywhere is a blocker, not a footnote.
- Measured per-call FFI overhead under 1µs.
- A written go/no-go decision recorded in this file, with the numbers behind it.

**Effort:** M — calendar-heavy, because profiling iterations are serial.

> **Decision point.** If M1 fails, the correct outcome is to ship lucene-rust as a
> standalone Rust library for reading Lucene indexes — which is genuinely useful and nearly
> done — rather than to proceed to M2–M6. That is what the gate is for.

---

## M2 — OpenSearch serving search from Rust

> Full detail, task breakdown and risks: [`docs/milestones/m2-opensearch-read-path.md`](milestones/m2-opensearch-read-path.md)

**Goal:** an OpenSearch node answers `_search` requests out of the Rust engine, over
Java-written segments, with clean fallback to Java Lucene for anything unsupported.

**Why now:** this is `PLAN.md`'s Phase 4 milestone and the first point where the project
produces user-visible value. The Rust side is already built; the entire gap is Java.

**Tasks**

- Build `opensearch-plugin/` for real: Gradle build, `EnginePlugin` implementation,
  per-platform native library packaged in the jar and loaded at startup.
- Binding layer over the existing 76 C-ABI functions. Evaluate Panama/FFM against JNI — the
  C ABI was deliberately designed to support both, and OpenSearch's JDK baseline allows FFM.
- Translate the OpenSearch query DSL subset onto the FFI query tree. **Any query outside the
  supported matrix falls back to Java Lucene per-query**, never errors — the hybrid path is
  what makes incremental adoption possible.
- Harden the boundary: `cargo-fuzz` over the FFI surface, handle-validation and
  use-after-close tests, and an assertion that every panic maps to an error code and never
  unwinds into the JVM.
- Wire the read-side lifecycle: reader refcounting and close determinism across the
  boundary, so mmap'd segments are released when OpenSearch expects.
- Confirm the M1 FFI overhead budget holds end-to-end, under real query load.

**Acceptance criteria**

- The OpenSearch REST search test suite passes on the Rust engine for the supported matrix,
  with fallback covering the rest.
- A fuzz and fault-injection run over the FFI surface produces zero JVM crashes and zero
  leaked handles.
- Killing and restarting a node with the Rust engine loaded recovers cleanly.
- End-to-end query latency improvement is consistent with M1's standalone measurement.
- A documented, published table of which queries run native and which fall back.

**Effort:** M–L — mostly Java and build/packaging work, little Rust.

---

## M3 — A Rust-written index that real Lucene can read

> Full detail, task breakdown and risks: [`docs/milestones/m3-write-path-proven.md`](milestones/m3-write-path-proven.md)

**Goal:** real Java Lucene opens a full, non-toy, Rust-written index with
`DirectoryReader.open`, passes its own `CheckIndex`, and returns hit lists and scores
identical to this port's searcher.

**Why now:** this converts the write path from "round-trips through our own reader" to
"interoperates with the reference implementation". Until it holds, every write-path claim in
`docs/parity.md` is conditional and no OpenSearch write-path work can be trusted. It is also
the milestone that closes the largest remaining scope gaps in the port.

**Tasks**

- **Close the reverse-verification gap at the current scope first.** Add
  `write_postings_fixture` and `VerifyPostings.java` for exactly what the writer supports
  today — single field, single block, `docFreq < 256`, freq-only. Expect this to find bugs;
  finding them at the small scope is far cheaper than at the large one.
- **Generalise the blocktree writer** past its proof-of-concept scope: multi-field `.tmd`,
  block splitting at `min/maxItemsInBlock`, floor sub-blocks, and multi-level `.tip` tries.
  The reader already handles all of these against real fixtures, so the reader is the spec.
- **Generalise the postings writer**: full 128-value `ForUtil`/`PForUtil` bit-packed blocks
  (not only the group-varint tail block), skip data, impacts, and `.pos`/`.pay` for
  positions, offsets and payloads. This removes `Error::DocFreqTooLarge` and
  `Error::UnsupportedIndexOptions` as reachable outcomes.
- **`VerifyIndex.java`** — the end-to-end reverse verifier: open a Rust-written multi-field,
  multi-segment index in real Lucene, run a fixed query set, and compare top-k doc IDs and
  scores against this port's searcher.
- Resolve the `.si` index-sort encoding divergence: today it is this port's own format,
  explicitly *not* verified against `Lucene99SegmentInfoFormat`. Either derive the true
  `SortFieldProvider` wire format from a Java-generated sorted-segment fixture, or record
  index-sorted segments as unsupported-for-interop in `parity.md`.

**Acceptance criteria**

- Real Lucene reads a Rust-written index with ≥3 fields, ≥100k docs, at least one term above
  `BLOCK_SIZE` (256) docs, and positions, offsets and payloads indexed.
- Real Lucene's own `CheckIndex` reports zero errors on that index.
- For a ≥50-query set spanning term, boolean, phrase and range, top-50 doc IDs match exactly
  and scores match within 1e-5 between Java Lucene and this port.
- `parity.md` no longer describes the postings/blocktree writer as narrowly scoped, or
  states precisely and truthfully what remains.

**Effort:** L — the largest remaining chunk of pure Lucene-format work.

---

## M4 — Write path hardened for production

> Full detail, task breakdown and risks: [`docs/milestones/m4-write-path-hardened.md`](milestones/m4-write-path-hardened.md)

**Goal:** the Rust `IndexWriter` is crash-safe, concurrent, and byte-interoperable with
Java's — indexes written by either engine fully usable by the other.

**Tasks**

- Sparse `IndexedDISI.writeBitSet` (currently deferred) and the remaining doc-values write
  kind.
- Postings and points reordering in sorted merges. Today
  `merge_sorted_stored_only_segments` reorders stored fields, doc values, norms and term
  vectors, but not postings or points.
- Concurrent indexing: multiple DWPTs, a real merge scheduler, and `IndexFileDeleter`
  file-lifecycle handling.
- Crash fuzzing: `kill -9` at randomised points during add/update/delete/commit/merge, with
  an assertion of recovery to the last durable commit every time.
- Random-op differential fuzzing against Java `IndexWriter`: feed the same operation stream
  to both engines and assert equivalent resulting indexes.

**Acceptance criteria**

- A 24-hour random-op and random-crash fuzz leaves an index that real Lucene's `CheckIndex`
  passes, every time.
- Java-written and Rust-written indexes are interchangeable in both directions across the
  full supported field matrix.
- No file-handle or memory growth over the soak.

**Effort:** L.

**Scope decision required — vector search.** Flat `.vec`/`.vem` storage plus brute-force KNN
landed; HNSW (`.vex`) does not exist. If OpenSearch k-NN is in the target matrix, HNSW is a
milestone of its own and must land before M5 — it is not a small addition. If it is not in
the matrix, record k-NN as Java-served and move on. **Decide this before starting M4**, since
it changes M5's scope materially.

---

## M5 — OpenSearch indexing from Rust

> Full detail, task breakdown and risks: [`docs/milestones/m5-engine-integration.md`](milestones/m5-engine-integration.md)

**Goal:** an OpenSearch shard fully served by Rust — both indexing and search.

**Tasks**

- Soft-deletes write side: `softUpdateDocument` and the retention merge policy. The read side
  (visibility) is already ported; the write side is required for peer recovery.
- Engine implementation: sequence numbers, local checkpoint, and commit user data exactly as
  `InternalEngine` produces them; refresh → NRT reader; flush → commit. The translog stays
  Java-side.
- **Segment replication first** — `PLAN.md` recommends it and it is materially simpler: only
  primaries index, and replicas reuse the M2 read path.
- Field handling parity for `_source`, `_id`, `_seq_no`, `_primary_term` and `_version`, plus
  a get-by-id fast path over FFI.
- Operational integration: memory accounting bridged to circuit breakers, stats APIs, slow
  log hooks, graceful shutdown, and panic → shard-failed rather than node-down.
- Aggregations fed from FFI doc-value cursors in batched columnar reads, keeping the
  aggregation framework itself on the JVM.

**Acceptance criteria**

- OpenSearch `:server` engine tests plus the full REST suite for index/search/get/delete pass
  on the Rust engine for the supported matrix.
- Peer recovery and segment replication work end-to-end between a Rust primary and replicas.
- A Rust-engine panic fails exactly one shard, and the node survives.

**Effort:** XL — the largest milestone, and the one most dependent on OpenSearch internals.

---

## M6 — Production candidate

> Full detail, task breakdown and risks: [`docs/milestones/m6-production-candidate.md`](milestones/m6-production-candidate.md)

**Goal:** a build a team would actually deploy, with a documented rollback path.

**Tasks**

- Multi-day soak with random restarts, under mixed index and search load.
- Nightly performance regression tracking, so M1's gate stays held rather than being a
  one-off measurement.
- Complete `docs/parity.md` for the supported matrix, and publish a supported-vs-unsupported
  feature table for operators.
- Upgrade and rollback story: how a cluster moves onto and off the Rust engine, given that
  backward-codecs are out of scope (force-merge is the escape hatch).
- Licensing and attribution audit — Apache-2.0 derivative work, `NOTICE` correctness.

**Acceptance criteria**

- 7-day soak: no index corruption, no memory or handle growth, no unexplained shard failures.
- M1's performance bar still met on the final build.
- A rollback from the Rust engine to the Java engine executed successfully on a test cluster.

**Effort:** M.

---

## Explicitly deferred beyond this roadmap

Carried from `PLAN.md`'s non-goals, restated so the boundary stays visible:

- **Backward-codecs.** Old segments stay on the Java engine until force-merged.
- **HNSW / native k-NN.** See the M4 scope decision.
- **Native aggregations.** Fed from FFI doc-value cursors; the framework stays on the JVM.
- **Join, grouping, taxonomy facets.** OpenSearch reimplements these as aggregations.
- **`luke`, `benchmark`, `demo`, `monitor`, `replicator`, `expressions`, `classification`,
  `spatial3d`, `spatial-extras`.**
- **Scoring pluggability** beyond BM25, constant score, and the similarity trait.

---

## On estimates

`PLAN.md` estimates 14–18 months for a 3–5 person team on the full port. This repo reached
its current state — 245 commits, ~128k lines of Rust, 2524 tests — in seven days of
AI-assisted work, so those calendar figures do not transfer. The effort sizes above
(S/M/L/XL) are therefore *relative*; the ordering and the gates are the parts worth trusting.

The dependency structure is the durable claim: **M0 unblocks everything. M1 decides whether
M2–M6 should be funded at all, and costs a benchmark rather than a Java plugin to answer.
M2 and M3 are independent and can run in parallel. M3 is the correctness precondition for
any OpenSearch write-path work.**
