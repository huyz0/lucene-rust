# M1 — The performance gate

> **Goal:** answer, with data, the question `PLAN.md` says must be answered
> before paying for the rest of the project — *is Rust search decisively
> faster than Java Lucene?*

| | |
|---|---|
| **Effort** | M — calendar-heavy, because profiling iterations are serial |
| **Depends on** | [M0](m0-ci-and-green-tree.md) |
| **Unblocks** | [M2](m2-opensearch-read-path.md), [M3](m3-write-path-proven.md) — or ends the project |
| **Status** | **delivered — gate verdict: FAIL** (see [`docs/benchmarks/verdict.md`](../benchmarks/verdict.md)) |

---

## Why this milestone exists

`PLAN.md` §4 states the premise plainly:

> The read-only milestone is the natural go/no-go checkpoint: if Rust search
> isn't decisively faster there, stop before paying for the write path.

and §3.5 makes it an invariant rather than an aspiration:

> a slower "faithful" port is a bug, and finding out early is the point of the
> phased structure.

Neither has ever been evaluated. No Rust-vs-Java comparison exists anywhere in
the repo. Criterion benches exist in `lucene-util`, `lucene-store` and
`lucene-codecs`, but they measure Rust against Rust — they can tell you a
change made a function faster; they cannot tell you whether the port is worth
building.

**This milestone is placed second, ahead of the OpenSearch integration, and
that is a deliberate departure from `PLAN.md`'s ordering.** `PLAN.md` puts the
decision at the end of Phase 4, after the JNI layer and the Java plugin.
But the quantity being measured — query CPU over identical segments — needs
neither. The read path is already complete and can open Java-written segments
today. Running the benchmark now buys the verdict for the price of a harness
instead of the price of a Java plugin.

If the answer is no, everything downstream changes, and it is far better to
learn that from a week of benchmarking than from three months of Java work.

---

## Scope

### In scope

- A repeatable, checked-in benchmark harness comparing lucene-rust against
  Java Lucene 10.5.0 on identical inputs.
- A realistic corpus and query mix.
- Profiling and closing whatever gaps the measurement exposes, including SIMD
  work where the profile justifies it.
- A microbenchmark of the FFI boundary against `PLAN.md`'s <1µs/call budget.
- A written verdict.

### Out of scope

- Write-path or indexing-throughput benchmarking. The gate is about search.
- OpenSearch-level benchmarking — that is [M2](m2-opensearch-read-path.md)'s
  confirmation step, and it measures a different thing (the whole stack, not
  the engine).
- Optimising anything the profile does not implicate. Speculative SIMD work is
  explicitly not part of this milestone.
- Multi-node, sharding, or network effects.

---

## Tasks

### T1.1 — Build the corpus

The `fixtures/data/` indexes are byte-level test fixtures, some of them a
handful of documents. They are useless for measurement — at that size
everything fits in L2 and the benchmark measures the harness.

Build a real corpus, written by **Java Lucene 10.5.0** so that both engines
read byte-identical segments:

- Target ≥5M documents with realistic term distributions. An English Wikipedia
  extract is the standard choice and is what Lucene's own nightly benchmarks
  use; a licence-clean alternative is fine as long as term frequencies are
  Zipfian rather than uniform.
- Index the field types the query mix exercises: an analysed text field with
  positions and offsets, a keyword field, a numeric point field, a numeric
  doc-values field for sorting, and a sorted-set doc-values field for facets.
- Produce two variants: **many small segments** (post-refresh, pre-merge) and
  **one force-merged segment**. These stress different code — per-segment
  fan-out versus raw decode — and a port can plausibly win one and lose the
  other.
- Do not check the corpus in. Check in the generator and a manifest recording
  document count, segment count, and index size, so a result is reproducible.

### T1.2 — Build the query set

A fixed, checked-in query file, sized so no single shape can dominate the
verdict:

| Shape | Why it is in the mix |
|---|---|
| Term, high / medium / low doc-frequency | The base case; separates decode speed from setup overhead |
| Boolean `AND` (conjunction) | Exercises the leapfrog/`ConjunctionDISI` path |
| Boolean `OR` (disjunction) | Exercises MAXSCORE pruning — the port's most performance-sensitive search code |
| Boolean with `minimumNumberShouldMatch` | A path where pruning is harder |
| Exact phrase, and sloppy phrase | Position decode, the `.pos` read path |
| Prefix / wildcard | Term enumeration through the blocktree, not just postings |
| Points range | BKD traversal |
| Doc-values range | The skip-index path |
| Sort by numeric field | `TopFieldCollector` plus doc-values reads |
| Sorted-set facet counting | Ordinal-heavy per-doc work |

Include both selective and unselective variants of the range queries — a
predicate matching 0.1% of documents and one matching 40% take different code
paths in both engines.

### T1.3 — Build the harness

Two runners emitting the **same JSON schema**, plus a comparison script:

- `benchmarks/rust-runner/` — a Rust binary, not a criterion bench. Criterion's
  statistics are built for micro-measurements; this needs sustained
  throughput.
- `benchmarks/java-runner/` — a Java program against Lucene 10.5.0.
- `scripts/bench-compare.sh` — runs both, joins on query id, emits a table of
  ratios plus the summary verdict.

Report per query: queries per second, p50/p95/p99 latency, CPU time, and total
hits (as a correctness cross-check — if hit counts differ, the timing is
meaningless).

**Methodology, and it matters more than the harness code.** These are the
standard ways a JVM-vs-native benchmark produces a fake win:

- **JIT warmup.** Java must run enough warmup iterations to reach steady state
  before timing starts. A cold JVM will lose by 5–10× and the number is
  worthless. Report the warmup iteration count in the output.
- **Page cache.** Run both engines against a warm cache, and separately
  against a dropped cache. The cold-cache case is largely an I/O measurement
  and both engines should be close; a large gap there indicates an mmap or
  `madvise` difference worth understanding.
- **GC.** Fix the JVM heap explicitly and record the collector. Do not let the
  Java side run with a heap so small it GC-thrashes, and do not give it one so
  large the comparison is unrealistic.
- **Thread count.** Measure single-threaded first — it isolates the engine.
  Then measure concurrent search, which exercises the port's rayon fan-out
  against Java's `ExecutorService`.
- **CPU pinning and frequency scaling.** Pin both runners to the same cores.
  Record whether turbo is enabled.
- **Cross-check recall.** Every query must return identical total hit counts
  and identical top-k doc IDs from both engines. A speedup obtained by
  returning fewer results is a bug report, not a benchmark result.

### T1.4 — Profile and close gaps

Flamegraph the Rust side (`perf` + `cargo flamegraph`) and the Java side
(async-profiler) on identical workloads, and compare where time actually goes.

Optimise **only what the profile implicates.** `PLAN.md` §3.5 nominates PFOR
decode, BKD compare loops, and bitset operations as the SIMD candidates, and
`crates/lucene-codecs/src/for_util.rs` is the obvious first suspect — but
nominate nothing before measuring. The point of doing this after the harness
exists is that every optimisation gets a before-and-after number.

Where SIMD is justified, follow the existing constraints: `std::simd` with
runtime feature detection, a scalar fallback retained for differential
testing, and `unsafe` confined to `lucene-util` per invariant #4 in
`AGENTS.md`.

### T1.5 — Microbenchmark the FFI boundary

`PLAN.md` §2 Phase 4 sets a budget of **<1µs per search call** of FFI
overhead. Measure it now, against the 76 existing `extern "C"` entry points —
specifically the ones on the hot path: `ffi_search_term_query_scored`,
`ffi_search_boolean_query_multi_segment_maxscore`, `ffi_results_copy`.

Measure the crossing itself: the same query executed through
`lucene_search::` directly, versus through the `ffi_*` wrapper, and the
difference is the boundary cost. This is a **leading indicator only** — it
measures the C ABI, not JNI or FFM marshalling, which
[M2](m2-opensearch-read-path.md) confirms end-to-end.

If the per-call cost is already near 1µs before a JVM is involved, that is a
design signal worth acting on in this milestone: look for per-doc crossings
that should be batched, and per-call allocations in the results path.

### T1.6 — Decide the `Weight`/`Scorer` question, with evidence

`docs/parity.md` records `search/Weight`, `search/Scorer` and
`search/BulkScorer` as **deferred, not started** — this port wires scoring
directly into query execution instead of building the trait hierarchy.

That is a defensible Rust-first choice (`PLAN.md` §3.5 point 2 explicitly
wants monomorphised per-doc loops rather than megamorphic virtual calls), and
it should stay deferred **unless the profile shows it costs something real**.
The specific thing to look for: whether the absence of a `Scorer` abstraction
prevents dynamic pruning from generalising — MAXSCORE is currently
implemented narrowly, on the single-term and `BooleanQuery` paths only.

Introducing the hierarchy is a large architectural change. Do it for a
measured reason, or not at all, and record which in this file.

### T1.7 — Write the verdict

A new `docs/benchmarks/` directory containing the raw results, the environment
description (CPU, kernel, JDK build, heap, corpus manifest), the flamegraphs,
and a short verdict document.

Then update `docs/roadmap.md` and this file with the outcome. The verdict is a
project artifact, not a note — M6 will re-run this harness to confirm the bar
still holds.

---

## Acceptance criteria

These are two different things, and the original draft of this file conflated
them. The milestone's goal is to *answer* a question; the gate's criteria decide
what the answer is. A FAIL verdict is a delivered milestone, not a failed one —
the "If the gate fails" section below exists precisely because that is a valid
outcome worth acting on.

### Delivery criteria — must all hold for M1 to be done

- [x] A repeatable, checked-in harness runs both engines over an identical,
      Java-written index and query set, and reports throughput and p50/p95/p99 —
      `scripts/bench-compare.sh`, `benchmarks/{rust,java}-runner`.
- [x] The harness cross-checks recall before comparing timings, comparing hit
      sets (order-insensitive) and top-1 scores within 1e-5, and reports
      equal-score tie reordering separately from genuine mismatches.
- [x] The corpus is reproducible from a checked-in generator plus a manifest,
      with no checked-in gigabytes — `scripts/bench-corpus.sh`, 5M docs.
- [x] Both the many-small-segments (15 segments) and force-merged (1 segment)
      variants are measured.
- [x] Per-call FFI overhead measured against the <1µs budget: **≈0 ns**, below
      noise. The one gate criterion that passes.
- [x] The `Weight`/`Scorer` question is decided **with evidence** — adopt a lazy
      skippable iterator; see the verdict's T1.6 section.
- [x] A written verdict exists in `docs/benchmarks/verdict.md`, with
      `environment.md` recorded alongside the numbers.

### Gate criteria — these decide PASS or FAIL, not delivery

**Result: FAIL.** Recorded here rather than left unticked, because an unticked
box reads as "not yet measured" when in fact it was measured and missed.

- [x] ~~**≥1.5× throughput**~~ — **FAILED: 0% of queries** (required ≥80%)
- [x] ~~No workload slower~~ — **FAILED: 20 of 20 slower**
- [x] ~~Identical hit sets and scores~~ — **FAILED: 19-20 of 20 mismatch**, all
      traced to the BM25 `avgdl` bug
- [x] ~~Same direction on both corpora~~ — **held**: both variants 0/20
- [ ] Reproduced on a second machine — **not done**, only one machine available
- [ ] **≥1.5× throughput** versus Java Lucene 10.5.0 on **≥80% of the query
      mix**, at identical recall, single-threaded, warm cache, force-merged.
- [ ] **No workload slower than Java** — `AGENTS.md` invariant #3 makes a
      regression a bug, not a footnote.
- [ ] Every query returns identical hit sets and top-1 scores within 1e-5.
- [ ] The segmented corpus shows the same direction of result as the merged one.
- [ ] The headline ratio reproduces on a second machine.

## The decision

**If the gate passes:** proceed to [M2](m2-opensearch-read-path.md) and
[M3](m3-write-path-proven.md), which are independent and may run in parallel.

**If the gate fails:** the correct outcome is to ship lucene-rust as a
standalone Rust library for reading Lucene indexes — a genuinely useful
artifact that is nearly complete today — rather than proceeding to M2–M6.
Stopping here is the gate working, not the project failing.

**If the gate is marginal** (a win, but under 1.5×, or uneven across shapes):
do not quietly proceed. The honest options are to spend a bounded, explicitly
time-boxed optimisation round and re-measure, or to narrow the target to the
query shapes that do win and re-scope M2 around them. Both are legitimate;
drifting forward without deciding is not.

---

## Risks and unknowns

- **Measuring the wrong thing** is the dominant risk. Every failure mode in
  T1.3 has produced a published, wrong native-beats-JVM benchmark at some
  point. Treat a surprisingly large win as a bug in the harness until proven
  otherwise.
- **Corpus licensing and size.** A multi-gigabyte corpus cannot be checked in,
  so reproducibility depends on the generator being deterministic and the
  manifest being accurate.
- **The result may be uncomfortable.** The value of the gate comes entirely
  from being willing to act on a negative answer. Deciding in advance what
  "fail" triggers — as this file does — is what makes that possible.
- **Hardware variance.** A single machine's result is one data point. Confirm
  the headline ratio on a second machine, ideally a different microarchitecture,
  before treating it as the verdict.

---

## Exit artifacts

- `benchmarks/rust-runner/`, `benchmarks/java-runner/`
- `benchmarks/corpus/` generator and manifest
- `benchmarks/queries.json`
- `scripts/bench-compare.sh`
- `docs/benchmarks/results-<date>.md`, flamegraphs, environment record
- `docs/benchmarks/verdict.md` — the go/no-go, with numbers
