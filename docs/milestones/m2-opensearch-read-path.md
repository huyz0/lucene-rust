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
| **Status** | not started |

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

- [ ] The plugin installs into an OpenSearch node and the node starts.
- [ ] The OpenSearch REST search test suite passes on the Rust engine for the
      supported matrix, with fallback covering the remainder.
- [ ] Every unsupported query **falls back and returns correct results** —
      zero errors attributable to an unsupported shape.
- [ ] Fuzz and fault-injection over the FFI surface: **zero JVM crashes, zero
      leaked handles**, every panic surfaced as an error code.
- [ ] Killing (`SIGKILL`) and restarting a node with the engine loaded recovers
      cleanly.
- [ ] Force-merging releases the superseded segment files — no reader leak
      pinning disk.
- [ ] End-to-end latency improvement is consistent in direction and rough
      magnitude with M1's standalone measurement.
- [ ] Measured FFI overhead stays under 1µs/call with the JVM in the loop.
- [ ] A published table of which query shapes run native and which fall back,
      committed to `docs/`.
- [ ] The plugin builds and its tests pass in CI on both linux-x64 and
      linux-aarch64.

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
