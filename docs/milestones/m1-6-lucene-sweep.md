# M1.6 — Lucene source sweep: parity gaps and hot-path optimisation

**Status:** Stages A, B, C, D1, E1 and the search-layer part of F delivered 2026-08-28; D2 and E2 open, and the two milestone-sized findings (block-tree navigation, lazy positions) filed rather than fixed. Verdict: [`docs/benchmarks/verdict-m1.6.md`](../benchmarks/verdict-m1.6.md). Findings: [`docs/sweep/findings.md`](../sweep/findings.md).
**Reference:** Apache Lucene `releases/lucene/10.5.0`, extracted from the clone at
`~/work/lucene` (that clone's checkout is `main`/11.0.0-SNAPSHOT — the sweep reads the
10.5.0 tag via `git archive`, never the working tree, because 11.0 has already changed
formats we are pinned to).

---

## Goal

Read this port's implementation **file by file against the Lucene source it ports**, and
for each file answer three questions with evidence:

1. **Parity** — does it miss anything Lucene does? (behaviour, format, edge case)
2. **Optimisation** — is the Rust doing what Lucene's Java does *efficiently*, covering
   SIMD, byte/struct layout and alignment, and zero-copy? Java has the JIT and the Panama
   Vector API; where Lucene vectorises, a scalar Rust transliteration is a defect, not a
   stylistic choice.
3. **Evidence** — is there a microbenchmark measuring *this component* against *Lucene's
   equivalent component*, rather than only the end-to-end query benchmark from M1?

Compatibility is not negotiable: every change must leave the differential suite
(`scripts/gen-fixtures.sh`, `scripts/verify-write-path.sh`, 2531 unit/fixture tests) green.
An optimisation that changes a single decoded byte is a bug.

### Why this is worth doing now

M1's gate stands at 1/20 queries reaching 1.5× of Java, and M1-e2e's profile came out
*flat* — largest single item 14.78%. A flat profile means there is no one mechanism left
to fix; it means the constant factor is spread across the whole decode stack. That is
exactly the shape a component-by-component sweep addresses and an end-to-end benchmark
cannot: end-to-end numbers cannot tell you that `ForUtil.decode` is 3× off when it is 9%
of a profile that is 9% everywhere.

Reconnaissance already found three concrete instances before the plan was written (all in
`for_util.rs`, the innermost decode loop in the engine):

- **Loop nest is the pre-10.2 order.** `split_ints` iterates `for i { for j }`. Lucene
  swapped to `for j { for i }` and its own comment says why: *"Process each shift level
  across all elements (better for vectorization)"*. Our order gives the inner loop a
  variable trip count and a stride-`count` write pattern — the two things that stop a
  vectoriser cold.
- **No SIMD at all.** Lucene ships `MemorySegmentPostingDecodingUtil`, which loads
  `IntVector`s straight out of the mapped segment and does the shift/mask lanewise. We
  have no equivalent and no dispatch layer to hang one on.
- **Per-call scratch.** `for_decode` declares `let mut tmp = [0u32; BLOCK_SIZE]` on every
  call — a 1 KiB zero-fill per 256-value block. Lucene's `ForUtil` holds `tmp` as an
  instance field and reuses it forever. `decode_slow` allocates a second one;
  `encode_generic` heap-allocates with `vec![]`.

None of those were visible from a profile. All three came from reading the two files side
by side. That is the case for the sweep.

---

## Method

Files are swept in **hot-path-first order**, because that is where an optimisation finding
is worth acting on and where a parity gap is most likely to have been papered over by a
single-segment fixture.

For each file the sweep produces a row in `docs/sweep/findings.md`:

| field | meaning |
|---|---|
| Rust file | the file swept |
| Lucene counterpart | the 10.5.0 source read against it |
| Parity findings | behaviour/format gaps, each either fixed or filed with a reason |
| Optimisation findings | each classified SIMD / layout / zero-copy / allocation / algorithmic |
| Measurement | the microbenchmark that proves the before/after, and the Java number it is compared against |

A finding is only closed by a **measurement**, not by an argument. A change that is
"obviously faster" and measures flat gets reverted — M1.5 already produced two of those
(header-only block skipping was 41% *slower*; WAND partitioning broke an invariant test
for a 2.1× on one query) and both were caught only because they were measured.

### Microbenchmark protocol

Rust: `criterion`, in the crate that owns the code.
Java: a plain warmup+timed harness under `benchmarks/micro/`, using the same
Lucene 10.5.0 jars `scripts/lib-lucene-jars.sh` already resolves, run with
`--add-modules jdk.incubator.vector` so Lucene's Panama paths are actually live (without
it Lucene falls back to `DefaultVectorizationProvider` and any comparison flatters us).

Lucene ships `PostingIndexInput` specifically so that posting decode can be benchmarked
from outside — its own javadoc says so — but `ForUtil` is package-private, so the Java
harness classes live in `org.apache.lucene.codecs.lucene104` to reach it. That is the
supported route, not a hack around encapsulation.

Both sides emit ns/op in the same TSV shape and are joined by a compare script, reusing
M1's load guard (`BENCH_MAX_LOAD`) — two M1 measurement rounds were thrown away to machine
load before that guard existed.

---

## Task list

Ordered. One at a time; review the diff before each commit.

### Stage A — the innermost decode loop

- **A1** `lucene-codecs/src/for_util.rs` vs `lucene104/ForUtil.java` +
  `PostingDecodingUtil` + `MemorySegmentPostingDecodingUtil`.
  Establish the Rust/Java microbenchmark pair for `decode(bitsPerValue)` across
  `bitsPerValue in 1..=32` first, so every later change in this file is measured.
- **A2** Fix the `split_ints` loop nest order to match Lucene's; measure.
- **A3** Remove per-call scratch buffers (`tmp` in `for_decode`/`decode_slow`, `vec!` in
  `encode_generic`); measure.
- **A4** Bulk word read: `split_ints` currently calls `read_u32_le()` `count` times, one
  bounds check and one `Result` per word, where Lucene issues a single `readInts`. Add a
  bulk primitive to `DataInput` with a `SliceInput` override; measure.
- **A5** Explicit SIMD for the shift/mask kernel, with runtime feature detection and a
  scalar fallback that stays the reference implementation. Only if A2–A4 leave a gap that
  measures.

### Stage B — postings

- **B1** `lucene-codecs/src/postings.rs` vs `Lucene104PostingsReader.java` — block decode,
  `LazyDocsCursor::advance`, impacts/skip levels, prefix-sum reconstruction.
- **B2** `lucene-codecs/src/postings.rs` positions/payloads vs the `.pos`/`.pay` reader.

### Stage C — the byte layer

- **C1** `lucene-store/src/data_input.rs` vs `DataInput`/`GroupVIntUtil` — group-varint
  decode, and whether the trait forces copies Lucene's `MemorySegmentIndexInput` avoids.
- **C2** `lucene-store/src/directory.rs` vs `MMapDirectory`/`MemorySegmentIndexInput` —
  mapping strategy, `madvise`, slice/clone cost.

### Stage D — term dictionary

- **D1** `lucene-codecs/src/blocktree.rs` vs `Lucene90BlockTreeTermsReader`/`SegmentTermsEnum`.
- **D2** `lucene-codecs/src/fst.rs` vs `util/fst/FST.java` — arc lookup, direct addressing.

### Stage E — packed/values

- **E1** `direct_reader.rs`/`packed_ints.rs`/`direct_monotonic.rs` vs `DirectReader`/
  `DirectMonotonicReader` — Lucene 10.5 vectorises parts of doc-values bulk decode
  (`DocValuesBulkDecodeSupport`/`PanamaDocValuesBulkDecodeSupport`); we have no analogue.
- **E2** `norms.rs`, `indexed_disi.rs`, `doc_values.rs`.

### Stage F — scoring

- **F1** `lucene-search/src/similarity.rs`, `field_norms.rs`, `collector.rs` vs
  `BM25Similarity`/`HitQueue` — score loop, norm table lookup, priority queue.
- **F2** `lucene-search/src/lib.rs` hot loops vs the corresponding `Scorer`s.

Stages G+ (write path, FFI, analysis) are lower value for optimisation but still owe a
parity read; they are scheduled only after A–F close.

---

## Acceptance criteria

**Delivery** (these decide whether the milestone is done):

1. Every file in Stages A–F has a row in `docs/sweep/findings.md` naming the Lucene source
   read against it, with findings listed and each one either fixed, measured-and-reverted,
   or filed with a stated reason.
2. Every Stage A–F file with a hot loop has a Rust microbenchmark and a Java counterpart
   measuring the same operation, with both numbers recorded.
3. The differential suite is green at every commit: 2531+ tests, `scripts/gen-fixtures.sh`,
   `scripts/verify-write-path.sh`, clippy clean on x64 and aarch64.
4. `docs/parity.md` rows are updated for every file whose status the sweep changes.

**Outcome** (these are measured and reported, not promised):

5. The measured Rust-vs-Java ratio for each benchmarked component, before and after.
6. A re-run of M1's end-to-end benchmark, reporting what the component work moved.

Criterion 5 is deliberately not "component X must reach parity". The sweep's job is to
find and measure; if a component turns out to be at parity already, that is a result worth
recording, and if one cannot reach parity without a redesign, that is a finding to file,
not a target to miss. M1 established this split: a FAIL verdict honestly measured is a
delivered milestone.


---

## Delivery status (2026-08-28)

### Swept and closed

| stage | file | outcome |
|---|---|---|
| A1-A4 | `lucene-codecs/src/for_util.rs` | 0.75x -> **2.30x** of Lucene. Three findings: loop nest order, per-call scratch, missing bulk word read. |
| A5 | (SIMD) | **Deferred with a measured reason** — the scalar kernel beats Lucene's Panama one on all 31 widths; explicit SIMD would mean relaxing `#![forbid(unsafe_code)]` and maintaining a second kernel forever. |
| B1 | `lucene-codecs/src/postings.rs` | 0.20x -> **1.69x** of Lucene on `nextDoc()`. `next_doc` was binary-searching for an offset that is always 1. Plus per-block scratch and lazy impacts. |
| C1 | `lucene-store/src/data_input.rs` | `read_u32s_le` added (Lucene's `readInts`), with a second `DataInput` backend in tests so the provided methods' default forms actually execute. |
| C2 | `lucene-store/src/directory.rs` | **Checked, no gap at the default.** Lucene's `MMapDirectory.readAdvice` initialises to `Optional.empty()`, so its per-file `madvise` hints are inert unless a caller installs `ADVISE_BY_CONTEXT`. `IndexInput.prefetch` has no equivalent here; unmeasurable against a warm page cache. |
| D1 | `lucene-codecs/src/blocktree.rs` | **Largest architectural divergence found.** Reader open is **135x slower than Lucene** (560 ms vs 4.2 ms on the 15-segment corpus) because the whole term dictionary is materialized at open. Filed, not fixed. |
| E1 | `lucene-codecs/src/direct_reader.rs` | 0.82x -> **1.82x** of Lucene, flat across all fourteen widths. It read one byte at a time where `DirectPackedReaderNN` does one wide load. |
| B2 | phrase positions | 0.04x -> **0.31x**. Per-document `Vec` allocation removed (`read_positions_flat`), and the redundant match pass for single-clause queries removed. Full laziness still open. |
| F1 | `lucene-search/src/{lib,collector,field_norms}.rs` | `ImpactsDISI`'s per-block bound, `TopScoreDocCollector`'s fast reject, dense one-byte norm fast paths, idf hoisted out of both boolean scorers. |
| — | `postings_writer.rs` | Doc-delta encoding choice corrected to Lucene's condition (P2). |
| — | multi-segment statistics | Phrase, dismax and boost clauses now get reader-wide idf; segmented recall **13 -> 0** mismatches. |

### Open, in priority order

1. **D1 — real block-tree navigation.** Reader open at 135x Lucene, and O(vocabulary)
   memory per segment. Blocks M2 and M5 independently of query speed, because a
   search engine reopens readers on every refresh. Milestone-sized: FST arc
   walking to a block, frame-based suffix scanning, lazy metadata decode.
2. **B2 — a lazy positions cursor.** The allocation half is fixed; positions are
   still all decoded before any document is known to be a phrase candidate.
   M1.5-sized.
3. **`tf_norm`'s second division** in the boolean scorers (see the verdict).
4. **E2 — `norms.rs`, `indexed_disi.rs`, `doc_values.rs`** not yet read against
   their Lucene counterparts. `norms` was partly covered via `field_norms`'s
   fast path; the other two are untouched.
5. **WAND clause partitioning.** Milestone-sized.

### Acceptance criteria, scored

| # | criterion | status |
|---|---|---|
| 1 | Every Stage A-F file has a findings row naming the Lucene source read against it | **partial** — A, B1, B2, C1, C2, D1, E1 and the F files are covered; E2 (`norms`/`indexed_disi`/`doc_values`) and D2 (`fst.rs`) are not. |
| 2 | Every swept file with a hot loop has a Rust microbenchmark and a Java counterpart | **met for what was swept** — `for_decode` and `postings_iter` both have paired harnesses in `scripts/bench-micro.sh`. The search-layer findings were measured end to end and by profile, not by a paired microbenchmark; there is no Lucene class to isolate for "the term scoring loop". |
| 3 | Differential suite green at every commit | **met** — 2539 tests, `verify-write-path` 13/13, `gen-fixtures` ok, clippy clean, at every commit. |
| 4 | `docs/parity.md` updated where the sweep changed a file's status | **met** |
| 5 | Measured Rust-vs-Java ratio per component recorded before and after | **met** — see the verdict. |
| 6 | M1's end-to-end benchmark re-run, reporting what moved | **met** — median 0.15x -> 0.34x, recall mismatches 13 -> 0 on the segmented corpus, gate still FAIL at 1/20. |

Criterion 1 is the one that is genuinely incomplete, and the milestone is not
claimed as done on that account. What is delivered is the method, the harness,
and the stages that carried the measured cost.
