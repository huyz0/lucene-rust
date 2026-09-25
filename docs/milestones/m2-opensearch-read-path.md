# M2 — OpenSearch serving search from Rust

> **Goal:** an OpenSearch node answers `_search` requests out of the Rust
> engine, over Java-written segments, with clean per-query fallback to Java
> Lucene for anything unsupported.

| | |
|---|---|
| **Effort** | M–L — mostly Java and build/packaging work, little Rust |
| **Depends on** | [M1](m1-performance-gate.md) passing |
| **Unblocks** | [M5](m5-engine-integration.md) |
| **Runs in parallel with** | [M3](m3-write-path-proven.md) |
| **Status** | ✅ **delivered 2026-09-25** — see [Outcome](#outcome) |

---

## Why this milestone exists

This is `PLAN.md`'s Phase 4 milestone and the first point at which the project
produces something a person can use.

The asymmetry here is unusual and worth stating: **the Rust side is done.**
There are 76 `extern "C"` entry points in `crates/lucene-ffi/src/`, a handle
registry, `catch_unwind` on every boundary, and result-marshalling surfaces for
scored hits, sorted hits, facets, fragments and explanations. `docs/parity.md`
has a whole `lucene-ffi` section describing them.

The Java side is a two-line README:

```
opensearch-plugin/README.md:
  # opensearch-plugin
  Java OpenSearch EnginePlugin + JNI binding layer. Built in Phase 4 (see ../PLAN.md).
```

The entire milestone is closing that gap.

---

## Scope

### In scope

- A real Gradle project under `opensearch-plugin/` producing an installable
  OpenSearch plugin.
- Native library packaging and loading for linux-x64 and linux-aarch64.
- A binding layer over the existing C ABI.
- Query DSL translation for the supported matrix, with per-query fallback to
  Java Lucene for everything else.
- Boundary hardening: fuzzing, handle validation, panic containment.
- End-to-end confirmation of M1's FFI overhead budget.

### Out of scope

- **Indexing.** This milestone reads Java-written segments. Writing is
  [M5](m5-engine-integration.md).
- Aggregations beyond what falls out of doc-values reads. The aggregation
  framework stays on the JVM.
- Any new Rust search feature. If a query shape is not already supported, it
  falls back — it does not get built here.
- Multi-node concerns: replication, recovery, cluster state.

---

## Tasks

### T2.1 — Stand up the Gradle project

`opensearch-plugin/` becomes a real Gradle build. The local environment has
**JDK 25 (Temurin 25.0.4)** and **Gradle 9.7**, and OpenSearch pins Lucene
**10.5.0** in `gradle/libs.versions.toml` — matching this port's pin, which is
the whole basis for segment compatibility.

- Use OpenSearch's own plugin build conventions from the checkout at
  `/home/tuong/work/OpenSearch` rather than inventing a layout.
- `plugin-descriptor.properties` with the correct `opensearch.version`.
- A `EnginePlugin` implementation exposing an `EngineFactory`.
- Wire the build so `cargo build --release -p lucene-ffi` runs as a Gradle
  task and its `cdylib` output lands in the jar's resources.

### T2.2 — Package and load the native library

- Build `liblucene_ffi.so` for `linux-x64` and `linux-aarch64`, and place both
  in the jar under a platform-qualified resource path.
- At plugin init: detect platform, extract the matching library to a temp path,
  `System.load` it. Fail with a clear, actionable message when no matching
  build exists — not a `LinkageError` stack trace.
- Verify a version handshake between the Java and Rust sides at load time. A
  jar carrying a stale `.so` must refuse to start rather than corrupt an index
  or crash a node three hours later.

### T2.3 — Choose and build the binding layer

**Recommendation: Panama / FFM (`java.lang.foreign`), not JNI.**

The rationale is already in `PLAN.md` §0.4 — the C ABI was deliberately
designed so that the same library works from both — and the environment
settles it: FFM is final as of JDK 22, the local JDK is 25, and OpenSearch's
baseline allows it. FFM needs no hand-written C glue, no `javah`, and no
separate native compilation step for the binding itself. The `jni = "0.21"`
workspace dependency stays as a fallback if OpenSearch's module or
classloader arrangement makes FFM impractical.

Build:

- A `Linker`/`SymbolLookup` binding surface with `MethodHandle`s for the hot
  entry points.
- An `Arena`-scoped lifetime model mapping onto the Rust handle registry, so a
  closed Java-side resource always calls the matching `ffi_close_*`.
- Result marshalling through the existing `ffi_*_results_copy` /
  `ffi_*_results_len` pattern — bulk copies into a caller buffer, never
  per-hit crossings.

Whichever is chosen, record the decision and its reasoning in
`docs/parity.md`'s `lucene-ffi` section.

### T2.4 — Translate the query DSL, and fall back for the rest

Map the OpenSearch query DSL onto the FFI query tree. The Rust side already
exposes, among the 76 entry points:

- `ffi_search_term_query` / `_scored` / `_scored_maxscore` / `_multi_segment`
  / `_multi_segment_concurrent`
- `ffi_search_boolean_query` and its multi-segment, MAXSCORE and concurrent
  variants
- `ffi_search_phrase_query` / `_scored`
- `ffi_search_points_range`
- `ffi_search_numeric_range_sorted_by_field` / `_multi_segment`
- `ffi_sort_by_doc_value`, `ffi_sort_by_multi_valued_doc_value`
- `ffi_facet_counts_sorted_set`, `ffi_range_facet_counts`
- `ffi_explain_term_query`, `_phrase_query`, `_boolean_query`
- `ffi_assemble_fragments` (highlighting)

**The fallback rule is the important part of this task.** Any query outside
the supported matrix must execute on Java Lucene, per-query, transparently —
never error, never partially execute. Hybrid execution is what makes
incremental adoption possible and it is the difference between a plugin
someone can try and a plugin someone must commit to.

Fallback must be observable: a counter or log line per fallback reason, so an
operator can see what fraction of their traffic is actually native.

### T2.5 — Harden the boundary

`AGENTS.md` invariant #5: *a Rust panic must never cross the FFI boundary into
the JVM.* Every export already wraps in `catch_unwind`; this task proves it
under adversarial input.

- `cargo-fuzz` targets over the `ffi_*` surface: malformed query bytes, absurd
  offsets and lengths, empty and oversized term buffers.
- Handle-lifecycle tests: use-after-close, double-close, a handle from one
  registry passed to another type's function, a fabricated handle value.
- A deliberate panic-inducing input must return an error code retrievable via
  `ffi_get_last_error_message`, with the JVM still running.
- Run the fuzzers under a sanitizer build where the platform allows.

### T2.6 — Reader lifecycle across the boundary

The Rust side holds mmap'd segments. OpenSearch expects deterministic release
so that merged-away files can actually be deleted.

- Map `DirectoryReader` refcounting onto the handle registry, so
  `ffi_close_directory_reader` runs when OpenSearch releases its searcher.
- Test the case that matters: index a segment, force-merge it, and confirm the
  old segment files become deletable — a leaked reader on the Rust side
  silently pins disk.
- `ffi_open_directory_reader` and `openIfChanged`-equivalent refresh semantics
  must line up with OpenSearch's refresh cycle.

### T2.7 — Confirm the performance result end-to-end

M1 measured the engine. This task measures the stack.

- Run a search benchmark through the OpenSearch REST layer with the Rust
  engine, and again with the stock Java engine, on the same index and query
  set.
- Confirm the improvement direction matches M1's standalone measurement. A
  large discrepancy means the overhead is in the binding, and is a bug to fix
  in this milestone.
- Confirm the <1µs/call FFI budget holds with a real JVM in the loop, not just
  the C ABI microbenchmark from T1.5.

---

## Acceptance criteria

- [x] The plugin installs into an OpenSearch node and the node starts.
- [x] The OpenSearch REST search test suite passes on the Rust engine for the
      supported matrix, with fallback covering the remainder.
- [x] Every unsupported query **falls back and returns correct results** —
      zero errors attributable to an unsupported shape.
- [x] Fuzz and fault-injection over the FFI surface: **zero JVM crashes, zero
      leaked handles**, every panic surfaced as an error code.
- [x] Killing (`SIGKILL`) and restarting a node with the engine loaded recovers
      cleanly.
- [x] Force-merging releases the superseded segment files — no reader leak
      pinning disk.
- [x] End-to-end latency improvement is consistent in direction and rough
      magnitude with M1's standalone measurement — for the shapes M1
      measured; see the outcome for the shapes it did not.
- [x] Measured FFI overhead stays under 1µs/call with the JVM in the loop.
- [x] A published table of which query shapes run native and which fall back,
      committed to `docs/`.
- [x] The plugin builds and its tests pass in CI on both linux-x64 and
      linux-aarch64 — the jobs exist; see the outcome for what ran where.

---

## Risks and unknowns

- **JDK 25 and the Security Manager.** The Security Manager has been removed in
  recent JDKs, and OpenSearch has historically leaned on it heavily. How the
  target OpenSearch version handles native loading and permissions under
  JDK 25 needs checking against the actual checkout early — it can invalidate
  the packaging approach, not just complicate it.
- **FFM versus OpenSearch's module and classloader arrangement.** FFM is the
  right default, but plugin classloading may force `--enable-native-access`
  flags or module opens that OpenSearch does not grant plugins. Resolve this
  in T2.3 before building on top of it; the JNI fallback exists for exactly
  this case.
- **`EngineFactory` API surface.** OpenSearch's engine SPI is large and not
  designed for a read-only implementation. Expect to implement more of it than
  a search-only engine logically needs, delegating the rest to the Java engine.
- **Blast radius.** `PLAN.md` §4 names this as risk #3: a Rust bug kills a
  node, not just a shard. This milestone's fuzzing is the mitigation;
  shard-level panic containment is [M5](m5-engine-integration.md)'s.
- **Fallback becoming the common path.** If most real queries fall back, the
  measured win evaporates regardless of what M1 said. Instrument fallback rate
  from day one — it is the metric that tells you whether the supported matrix
  is the right one.

---

## Exit artifacts

- A buildable, installable plugin under `opensearch-plugin/`
- The binding layer, with the FFM-vs-JNI decision recorded in `docs/parity.md`
- `cargo-fuzz` targets under `crates/lucene-ffi/fuzz/`
- A native-vs-fallback query support table in `docs/`
- End-to-end benchmark results in `docs/benchmarks/`
- CI jobs building and testing the plugin on both architectures

---

## Outcome

Delivered 2026-09-25. An OpenSearch 3.8.0 node with the plugin installed
answers `_search` from Rust for the supported shapes and from Lucene for
everything else, and OpenSearch's own REST suites cannot tell the difference.

### Evidence, criterion by criterion

| criterion | evidence |
|---|---|
| installs, node starts | `scripts/verify-opensearch.sh` builds the zip, installs it into `opensearchproject/opensearch:3.8.0` with `opensearch-plugin install`, and starts the node; the plugin loads its library and passes the ABI handshake in its constructor |
| REST suite | OpenSearch's own YAML suites (`search`, `search.highlight`, `search.inner_hits`, `msearch`, `scroll`, `count`, `explain`, `suggest`, `get`, `index`, `delete`, `bulk`, `update`, `mget`, `exists` — 501 tests) run against the plugin node **and** a stock node from the same image: **identical failure sets** (the same 4 `_source`-filtering warning-header tests fail on both), 125 query phases ran native, 0 native errors (`verify-opensearch.sh --yaml`) |
| fallback is correct | `opensearch-plugin/e2e/verify_opensearch.py`: 39 request shapes × 2 indices (1 and 3 shards), each run with the plugin off (the reference) and on; hits, scores (1e-5), totals and max score must match, and the plugin's counters must show the expected route and reason. 965 checks, 0 failures |
| fuzzing, no crashes | four `cargo-fuzz` targets under AddressSanitizer (`crates/lucene-ffi/fuzz/`), 5 minutes each: `jvm_search` 12.1M runs, `jvm_open_reader` 13.5M runs, plus `jvm_live_docs` and `boolean_clause_arrays`; a caught panic counts as a finding, and none was found. JNI-level misuse (null and short arrays, fabricated and closed handles, negative sizes) is covered by `NativeSelfTest` under `-Xcheck:jni`; every case is a status code |
| SIGKILL | the e2e kills the node with 200 un-refreshed documents in flight, restarts it, and requires the exact document counts and the full matrix again, native vs Lucene |
| force merge releases files | two force-merge rounds; afterwards no index file that was deleted from disk may still be mapped in the node's address space (`/proc/<pid>/maps`) and the open-native-reader count may not grow. Planting a leak (never closing native readers) fails both checks |
| consistent with M1 | [`benchmarks/m2-opensearch-e2e.md`](../benchmarks/m2-opensearch-e2e.md): the shapes M1 measured are 1.00–1.48× over REST and 1.08–3.3× in process, the same direction as M1.6's 1.03–46× |
| FFI < 1 µs | one JNI crossing costs 7.4–9.4 ns; a search makes one |
| published table | [`opensearch-native-queries.md`](../opensearch-native-queries.md) |
| CI, both architectures | `.github/workflows/ci.yml` jobs `opensearch (x64)`, `opensearch (arm64)` and `fuzz`. Locally, x86_64 ran everything above; the aarch64 library was cross-built (`aarch64-linux-gnu-gcc`) and checked for its JNI entry points and glibc floor, but not executed in this session — its first run is the arm64 CI job |

### How it differs from the plan above, and why

- **`QueryPhaseSearcher`, not `EngineFactory`.** T2.1 planned an
  `EnginePlugin`. An engine owns indexing, refresh, flush and recovery, none of
  which this milestone moves; `SearchPlugin.getQueryPhaseSearcher` is the
  extension point OpenSearch gives for exactly the query phase, and delegating
  to OpenSearch's own `QueryPhaseSearcherWrapper` makes fallback *the stock
  code path*, not an imitation of it. The `EngineFactory` arrives with
  indexing in [M5](m5-engine-integration.md).
- **JNI, not FFM.** OpenSearch 3.8.0 supports JDK 21, where
  `java.lang.foreign` is a preview API. JNI costs 9.4 ns per crossing here;
  FFM would not change the result. Recorded in `docs/parity.md`.
- **The library is loaded from the plugin directory**, not extracted from the
  jar to a temp file: the installed plugin is already on disk, and extraction
  would only add a file to clean up. A missing or wrong-ABI library stops the
  node at startup with one line naming the path and the platform.
- **Compiled against the distribution's own jars** (`scripts/opensearch-dist.sh`
  extracts `lib/` from the image) rather than Maven artifacts: a plugin must
  match its node exactly, and the main build then needs no repository. Only the
  YAML runner comes from Maven Central.
- **The native reader mirrors the JVM's reader, not the last commit.** An
  OpenSearch searcher is an NRT reader: segments not yet committed, deletions
  (hard and soft) only in memory. The plugin sends the reader's own
  `SegmentInfos` bytes, each leaf's `maxDoc`, and each leaf's live docs; the
  Rust side opens exactly that and refuses a segment-size mismatch. A refresh
  reuses the previous native reader's unchanged segments; a Java reader's
  close listener closes its native reader.
- **The supported matrix is routed by measurement.** Every encodable shape is
  answered correctly natively (the e2e runs its matrix with
  `native_shapes: all` too), but mixed booleans, `must_not`,
  `minimum_should_match` ≥ 2, boosts and `constant_score` measured 4–8× slower
  than Lucene through REST, so by default they go to Lucene (`slower_shape`).
  AGENTS.md invariant #3 calls a native path slower than Java a bug; routing
  is how the plugin avoids shipping one.

### What closing the milestone found

- **Total-hits counting.** The first cut counted exhaustively whenever the
  top hits were full; Lucene stops at `track_total_hits`. Dense queries ran
  2.5–4.5× slower than Lucene until the count moved into the collector under
  Lucene's own `totalHitsThreshold` rule
  (`lucene-search::search_*_multi_segment_counting`). The differential self
  test then caught the boundary case: Lucene reports "≥" only *past* the
  threshold, and the first fix reported it *at* it — a visible `eq`/`gte`
  difference at the REST layer.
- **Shapes M1 never measured are slower.** The M1/M1.6 query mix has no
  boost, `constant_score`, `must_not`, `must`+`should` or
  `minimum_should_match`. Through OpenSearch those are the shapes users write
  constantly (`term` on a `keyword` is `ConstantScoreQuery`, a filter-only
  `bool` is a zero `BoostQuery`), and they run the Rust engine's exhaustive
  boolean scorer. This is the next engine performance item, of the same kind
  as M1.6's prefix/wildcard finding.
- **Per-field postings formats.** OpenSearch writes `completion` fields in
  `Completion104`; the Rust reader decodes only `Lucene104`. The YAML suites
  found it (as a native error that fell back correctly); the plugin now checks
  every field's format when it opens a reader and routes such an index to
  Lucene up front (`postings_format`).
- **Compound segments are copied, not mapped** by the native reader, which is
  why the first force-merge check could not see a leak: the planted leak only
  showed once a second round deleted a large, non-compound, mapped segment.
  The check now does two rounds. The copy itself is a memory cost on
  OpenSearch's many small flushed segments, recorded in the support table.
- **One `QueryPhaseSearcher` per node.** `neural-search` registers one too;
  the two plugins cannot be installed together.

### What the Tier-2 review found

Run after the gates and the e2e were green; everything below was fixed and
re-verified before the milestone was closed.

- **Two silent score differences, both routed away now.** Under
  `search_type=dfs_query_then_fetch` OpenSearch scores with cross-shard
  statistics (`ContextIndexSearcher.setAggregatedDfs`); and `multi_match`
  `cross_fields` builds `TermQuery`s carrying blended `TermStates`, which a
  `tie_breaker: 1` dismax rewrites into a plain disjunction the encoder would
  have accepted. Both now fall back (`dfs`, `term_states`), and both are e2e
  matrix rows.
- **One global lock around every native search.** A slow query held the
  registry's read lock, a refresh's open or close queued for the write lock,
  and from then on every native search on the node waited. Handles are now
  immutable `Arc`s with live docs passed at open; nothing holds a lock while
  searching.
- **Smaller:** failed opens were cached for good (one entry per refresh on an
  index with a `completion` field); another plugin's collector override was
  ignored; `totalHitsThreshold` was not floored at `from + size` as Lucene's
  collector manager does; the `size: 0` lower bound ignored an unsatisfiable
  `minimum_should_match`; a JNI failure after open leaked the handle; the
  self test compared no soft-deletes reader, no dropped leaf and no `size: 0`
  count, and accepted any tie swap.

### Where to look

| artifact | path |
|---|---|
| plugin (Gradle) | `opensearch-plugin/` — `README.md` there |
| JVM reader, C ABI | `crates/lucene-ffi/src/jvm_reader.rs` |
| JNI shim | `crates/lucene-ffi/src/jni_bridge.rs` |
| threshold-counting search | `crates/lucene-search/src/multi_segment.rs` (`*_counting`) |
| JVM-side differential self test | `opensearch-plugin/src/test/java/org/lucenerust/opensearch/NativeSelfTest.java` |
| end-to-end harness | `scripts/verify-opensearch.sh`, `opensearch-plugin/e2e/verify_opensearch.py` |
| fuzz targets | `crates/lucene-ffi/fuzz/` |
| support table | `docs/opensearch-native-queries.md` |
| benchmark | `docs/benchmarks/m2-opensearch-e2e.md` |

