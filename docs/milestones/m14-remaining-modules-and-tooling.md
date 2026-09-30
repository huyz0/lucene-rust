# M14 — The remaining modules and the tooling

> **Goal:** the last modules of the Lucene 10.5.0 distribution -- non-default
> codecs, expressions, classification, replication, `misc`, `sandbox` -- plus
> Rust counterparts of the test framework, `luke`, the demo and the
> benchmark suite. At the end, `check-port-inventory.py`'s allowlist is empty.

| | |
|---|---|
| **Effort** | L |
| **Depends on** | [M7](m7-core-complete.md)–[M13](m13-search-application-modules.md) |
| **Unblocks** | "fully ported", as the inventory gate defines it |
| **Status** | not started |

---

## Why this milestone exists

After M7–M13 what is left is either self-contained or tooling:

| Module | Files | Lines of code | Note |
|---|---|---|---|
| `codecs` | 95 | 19,242 | `SimpleText`, bloom filters, `BlockTreeOrds`, `UniformSplit`, memory postings |
| `expressions` | 19 | 2,697 | JavaScript expressions compiled to JVM bytecode |
| `classification` | 19 | 1,961 | k-NN and naive Bayes text classifiers |
| `replicator` | 13 | 1,526 | index and taxonomy replication |
| `misc` | 39 | 4,107 | `IndexSplitter`, `HighFreqTerms`, sorting and store utilities |
| `sandbox` | 113 | 11,849 | whatever M10 did not need |
| `test-framework` | 237 | 60,830 | random indexing, asserting codecs, mock analyzers |
| `luke` | 163 | 15,251 | Swing index browser |
| `demo` | 20 | 2,585 | indexing and search examples |
| `benchmark` | 128 | 9,200 | the `byTask` benchmark framework |

---

## Scope

### In scope

- `codecs`, `classification`, `replicator`, `misc`, and the rest of `sandbox`,
  ported and differentially tested like everything else.
- `expressions`: the same language and results, compiled to a Rust
  closure tree or bytecode interpreter instead of JVM bytecode.
- **Test framework**: the pieces that make Lucene's own tests valuable,
  as Rust test utilities -- `RandomIndexWriter`, `MockAnalyzer`,
  `MockDirectoryWrapper` (fault injection), asserting codecs and readers,
  `LuceneTestCase`'s randomized configuration -- and a port of the
  highest-value Lucene test classes on top of them.
- **Tooling equivalents**, as command-line tools rather than ports of Java
  UIs: an index inspector covering `luke`'s views, the demo's indexing and
  search examples, and `byTask`-style benchmark scripts on the existing
  `bench-micro`/`bench-compare` harness.

### Out of scope

- `luke`'s Swing UI and `benchmark`'s Java-specific plumbing, recorded as
  `not needed` with the Rust tool that covers each capability.

---

## Tasks

- **T14.1** — The non-default codecs.
- **T14.2** — `expressions`.
- **T14.3** — `classification`, `replicator`, `misc`, `sandbox`.
- **T14.4** — Test framework utilities, then a first batch of ported Lucene
  tests chosen by the bugs they historically caught.
- **T14.5** — Index inspector CLI, demo, benchmark scripts.
- **T14.6** — Empty the allowlist; publish the final `parity.md`.

## Acceptance criteria

- [ ] Every ported module passes Java-fixture differential tests.
- [ ] Real Lucene reads indices written with each non-default codec.
- [ ] `expressions` evaluates Lucene's expression test corpus to the same
      doubles.
- [ ] `check-port-inventory.py` passes with an **empty allowlist**: every
      class of the 10.5.0 distribution is `ported` or `not needed` with a
      reason.

## Risks and unknowns

- **The test framework is open-ended.** Lucene has tens of thousands of
  tests; T14.4 ports the infrastructure and a chosen batch, not all of them.
  The acceptance bar is the infrastructure plus the batch, stated up front.

## Exit artifacts

- `docs/parity.md` with no `deferred` rows
- The inspector, demo and benchmark tools under `tools/` or `benchmarks/`
