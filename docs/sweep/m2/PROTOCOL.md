# M2 full-port sweep protocol

Goal: file-by-file, function-by-function comparison of every ported Rust file
against its Java counterpart in **`/home/tuong/work/lucene-10.5.0`**, to
find *every* divergence: missing ports, different algorithms, different data
types/widths, different constants, different edge-case handling, and
performance/memory divergences.

## Which Java tree to read (read this first)

**Compare against `/home/tuong/work/lucene-10.5.0`, NOT `/home/tuong/work/lucene`.**

`/home/tuong/work/lucene` is a git checkout sitting on `main` (2026-08-26),
which is 4574 commits past the version this port pins. AGENTS.md pins Lucene
**10.5.0**, matching OpenSearch's `gradle/libs.versions.toml`, and
`scripts/lib-lucene-jars.sh` downloads 10.5.0 jars -- so every fixture and
verifier is ground truth for 10.5.0 while the working tree's *source* is not.

Between the two: **1261 files and ~62k lines differ**, including 32 classes
this port has ported -- among them `BM25Similarity`, `TieredMergePolicy`,
`ForUtil`, `Lucene104PostingsReader`/`Writer`, `Lucene103BlockTreeTermsReader`/
`Writer`, `IndexedDISI`'s neighbours, `Lucene90DocValues{Producer,Consumer}`,
`BKDReader`/`Writer`, `HnswGraphBuilder`, `RegExp`, `BooleanQuery`,
`BooleanScorer`, `PhraseQuery`, `IndexWriter`, `CheckIndex`, `SegmentInfos`,
`LZ4`, both compressing writers, `ReadersAndUpdates`, `OrdinalMap`,
`UnifiedHighlighter`, `QueryParserBase`, `CodecUtil` and `DataInput`.

Batch c14 discovered this the hard way: `main`'s `ReadersAndUpdates` writes
doc-values updates as overlay deltas with a fold-to-dense compaction, an
entirely different design from 10.5.0's. Porting from the working tree would
have produced a second invented format -- the exact defect that batch existed
to remove.

`/home/tuong/work/lucene-10.5.0` is a plain extraction of the
`releases/lucene/10.5.0` tag (`git archive`, so the user's repo was not
touched). Use it for all source reading. Keep using the real 10.5.0 **jars**
for fixtures, which is what `scripts/gen-fixtures.sh` and
`scripts/verify-write-path.sh` already do.

## Rules for a sweep batch

1. **Find the Java counterpart first.** Use `docs/parity.md`, then grep
   `/home/tuong/work/lucene/lucene/**/src/java` for the class. Record the exact
   Java path(s) you compared against. If no Java counterpart exists (glue,
   FFI, Rust-only helpers), say so explicitly — do not invent one.
2. **Compare method by method.** For each Rust `fn`, name the Java method it
   corresponds to and state: identical / divergent / not-in-Java / missing.
   Enumerate the Java methods that have **no** Rust counterpart too.
3. **Classify every divergence** as one of:
   - `CORRECTNESS` — wrong output, wrong bytes, wrong scores, wrong edge case.
   - `MISSING` — a Java behaviour/method/branch that is simply not ported and
     that a caller could reach.
   - `PERF` — same result, different cost (time or memory).
   - `INTENTIONAL` — deliberate, justified Rust-idiom or scope divergence.
4. **Fix `CORRECTNESS` and `MISSING` without asking.** Add or extend tests for
   each fix. Prefer a Java-fixture differential test (see the
   `differential-testing` skill) when it is a file-format concern.
5. **For `PERF`, reason explicitly about better/worse than Java** — algorithmic
   complexity, allocation count, branch/bounds-check cost, cache behaviour. If
   the Rust is worse, fix it if the fix is contained; otherwise record it with a
   measurement. Back non-obvious claims with a microbenchmark (criterion benches
   live in `crates/*/benches/`).
6. **Gate before you finish**: `cargo fmt --all`, `cargo clippy --all-targets
   --all-features -- -D warnings`, `cargo test -p <crate>` must pass for the
   files you touched.

## Output

Write `docs/sweep/m2/<batch-name>.md` with:

- A `## <rust file>` section per file, giving the Java counterpart path(s).
- A method-correspondence table.
- A numbered finding per divergence: `[CORRECTNESS|MISSING|PERF|INTENTIONAL]`,
  what Java does, what we do, the consequence, and the resolution (fixed +
  where, or recorded + why not).
- A short `### Verdict` per file: swept-clean, or the open items.

Return to the caller ONLY a compact summary: files swept, count of findings by
class, the fixes applied (one line each), and anything left open. Do not paste
code or file contents back.
