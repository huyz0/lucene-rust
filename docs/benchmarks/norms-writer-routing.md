# Installed asserting norms writer

The differential replay and focused failure tests pass. Six paired measurements
resolve a Rust advantage for flush commits; merge commit differences remain
inside measured noise. The full repository gate passes, with every touched
production file above 95% line coverage. Final semantic review passes without
remaining findings.

`AssertingNormsCodec` installs the existing asserting norms consumer in the
native `Lucene104` writer graph. Each field supplies a repeatable dense or
sparse column. The adapter validates document order and bounds, replays the
native consumer's statistics, sparse-document and nonconstant-value passes,
and closes its delegate twice as Lucene's asserting consumer does. Its lower
consumer retains owned field buffers until close, when it writes the native
`.nvm`/`.nvd` pair. A timed operation must therefore include close to measure
real encoded output.

This is a writer-only installation. Native readers and the `Lucene104` codec
name remain fixed. Whole AssertingCodec composition, asserting reader
installation and IOContext routing remain outside this slice.

## Differential corpus and provenance

`GenNormsWriterRouting` uses real Lucene 10.5.0 `IndexWriter` sessions with an
SPI-registered `NormsRoutingOracle` FilterCodec. Its unique fixture codec name
allows readers to reopen the installed oracle; the reader implementations are
native Lucene104 components. The actual `AssertingNormsFormat` wraps a native
tracing delegate, inserted through its pinned-version private final instance
field. The oracle does not emulate the asserting checks or construct expected
norm bytes.

Six cases represent Rust's legacy fast, legacy general and explicit-document
flush paths, each sorted and unsorted. General-path documents enable term
vectors on one field. Each case commits six documents in two batches of three,
reopens and appends two more, deletes `b`, `e` and `g`, then merges to one
segment. Fields cover dense constant norms, dense varying norms, sparse
presence, present-empty text with a real zero norm, omitNorms, a field with no
surviving values after merge, and an explicit-only field absent from earlier
source segments. Unique numeric sort keys make physical document order
observable.

The manifest records exact segment field names, numbers, indexing options and
norm flags; physical stored IDs and live bits; norm presence and values; and
raw BM25 score bits for `TermQuery(varying:x)`. A missing norm is distinct from
a present value of zero. Readback checks norm presence and values against the
input texts before the generator writes its manifest.

Writer traces preserve consumer open, add-field, fresh producer acquisitions,
actual iterator/value calls, native add-field and double native close. The
recording iterator forwards native bulk `intoBitSet` and `docIDRunEnd` calls,
so tracing does not replace a sorted sparse pass with a synthetic scalar loop.
Native reader acquisitions are separate `READ` provenance rows. Those rows
form a sorted multiset with duplicates because reader-pool teardown order is
not a writer-consumer contract. They contain deterministic segment names and
no random segment IDs. Writer trace ordering is retained.

Java reopens a `NormsProducer` after writing norms and obtains a merge instance
for postings. Rust retains dense impact columns and supplies them directly to
its postings writer. Missing norms use an internal impact-column filler of
`1`; this is not a present norm and must never reach sparse norms output.
Native producer open/get/merge-instance/close counts, file-read timing,
iterator reuse and related failure timing therefore cannot establish a
cross-engine callback equivalence. Exact persisted norm presence/values,
document mappings and query results are the observable contract. The Java
producer traces remain provenance rather than fabricated Rust callbacks.

## Correctness evidence

The six-case replay compares the complete committed snapshots and each
configured consumer's input columns, including multiplicities. It independently
checks norms close before postings/doc-values selectors and stored-fields close
before merge norms. Resetting the codec restores native writes. A larger
270-document case exercises complete 128-posting blocks and tails, commits 21
deletions, merges 249 survivors and compares native/asserted norms and raw score
bits; CheckIndex must execute and pass the varying field's impacts check.

Failure regressions cover errors before opening, before/after add, after native
close, competing add/close errors, partial native constructor outputs, native
output close errors, failed cleanup, merge failure and retry. Collision tests
preserve a preexisting merge candidate and all source/commit bytes while removing
only a successfully created mate. The helper's units separately prove its
created-output ownership, including a factory rename of a preexisting file.
Ordinary flush recovery additionally invokes the writer's existing orphan
cleanup, which can reclaim unrelated unreferenced files. Callback panic recovery
and errors in later sibling formats retain their existing caller scope.

The first replay failed because reopened explicit writers reassigned old field
numbers when a late field appeared. Persistent field-number reservations now
preserve the Java snapshot, including source conflicts and imported fields.
Source schemas also remain checked after ordinary source removal. The facade
retains its configured registrations and field IDs across `delete_all`, while
discarding source-only reservations; Java clears its field-number map entirely.
Because this facade keeps a rolled-back writer usable, rollback preflights the
restored sources and reinstates their reservations. A schema registered after
`delete_all` can conflict with those sources. In that case `try_rollback`
returns an error before changing state, while the void `rollback` leaves the
current state unchanged. Java closes the writer on rollback instead.
The full-block test initially queued deletes without committing before a
committed-segment force-merge; its setup now commits those deletes explicitly.
The collision regression initially contradicted ordinary flush orphan cleanup;
it now checks the narrower helper ownership guarantee through merge recovery.

## Workload and timing boundary

`scripts/bench-norms-writer-routing.sh` pairs
`NormsWriterRoutingMicro.java` with `norms_writer_routing_micro.rs`. Both use
1,024 documents with stored indexed IDs. Rust uses its explicit-document API;
legacy fast/general correctness cases are not timed. Both engines receive the
same four tokenized text fields. Only the single word `x` is used, repeated to produce these lengths:

| Field | Per-document values |
| --- | --- |
| `constant` | Present in every document, length 1 |
| `varying` | Present in every document, length `doc % 5`, including zero |
| `sparse` | Missing for multiples of 3; otherwise zero for multiples of 5 or length 3 |
| `vanish` | Present only for multiples of 7, length 2 |

Default BM25's small-length norm encodings equal these lengths. Both engines
use their standard analyzer, in-memory directories, noncompound output, a
merge factor of 1,000, a 2,048-document flush limit and disabled RAM
auto-flushing. The Java merge scheduler is serial; the Rust force-merge call
completes before the timed commit returns. No sorting is configured in this
benchmark; sorted correctness belongs to the differential corpus.

The flush case buffers all 1,024 documents, then times only `commit`. Both
engines analyze documents and compute their per-document norms during corpus
preparation, outside the interval. The interval includes assembling buffered
norm columns, native or asserting consumer work, close/output, sibling format
work and commit publication. A guard requires zero `.nvm`/`.nvd` files before
the timer starts.

The merge case prepares eight committed segments of 128 documents, deletes all
multiples of 7 and commits those deletions before timing. That leaves 877 live
documents. Preparation verifies eight source segments and 16 source norms
files. The interval covers `forceMerge(1)` followed by `commit`, including
source reads, mappings, norms output/close, sibling formats and publication.
All remaining documents lack `vanish`, so the merged schema retains a norms
field whose iterator has no values.

Corpus allocation and analysis, source commits, deletion queuing and commit,
readback, integrity checks and writer/directory teardown are excluded. These
are whole flush/merge commit measurements; they do not isolate assertion-loop
cost, norms encoding alone, full document indexing throughput or search speed.

## Result checks and memory scope

Every iteration opens the completed index through native readers and checks
one leaf, expected maxDoc/numDocs, every stored ID and its physical order, and
every field's exact norm presence/value. Both engines checksum opened index files
through the reader's integrity check. A digest incorporates each stored ID,
then the presence flag and value for each of the four fields. Presence is
included separately, so a missing norm cannot masquerade as zero.

Each required result row contains the observed digest, document count, field
count and the sum of the committed segment's `.nvm`/`.nvd` sizes. The paired
report requires matching rows for all four timing cases in every repetition.
Byte totals exclude stale merge-source files, other index formats and total
index size. They measure encoded lengths, not byte-for-byte cross-engine
identity. Reader and fixture checks supply semantic evidence independently.

Before measurements, the asserted Rust writer created a filesystem index after
the deleted merge. Default Java Lucene104 readers opened it and verified digest
`308099040984830393`, 877 documents, four fields and 2,909 norm-file bytes.

Shallow configuration rows cover the adapting Rust codec's static size and
the Java adapting codec, anonymous norms-format wrapper and asserting-format
objects. They exclude referenced native codec/format objects, producer and
consumer graphs, retained field buffers, impact columns, directory state,
allocator/runtime overhead, peak heap and RSS. These scoped values cannot
establish whole-writer memory parity. Encoded-file totals are reported
separately from shallow configuration sizes.

## Sampling and reproduction

Run through the repository's capped test wrapper after the differential
baseline passes:

```sh
scripts/docker-test.sh scripts/bench-norms-writer-routing.sh
```

The script defaults to 1,000 ms accumulated operation time for warmup and
1,500 ms for measurement, per case and process. Optional first and second
arguments replace those budgets in milliseconds. Six repetitions alternate
Rust/Java and Java/Rust process order. Cases within each process keep the same
order: default flush, asserted flush, default merge, asserted merge.
Preparation and readback make wall time longer than those timing budgets.
Even a zero warmup argument performs one preparation/operation/readback
iteration.

The harness uses the capped Rust 1.97.1/JDK 21 environment, CPU affinity 0 and
1 by default (`BENCH_PIN` can override it), and a fixed 128 MiB JVM heap. A
load guard checks process starts, waiting up to 300 seconds for the one-minute
load average to fall to 1.5; it does not monitor an entire sample. Java is not
launched with the incubator Vector API module. Excluded work, JVM warmup/GC,
fixed case order and temperature can still affect measurements.

The shared reporter takes per-engine medians and reports repetition spread
and its estimated noise floor. Java/Rust ratios above 1.0 favor Rust. Ratios
inside measurement noise do not resolve an advantage. The asserted/default
contrast within either engine does not isolate wrapper overhead because it
also includes control-flow, allocation, JIT and ordering effects.

## Results

Measured on October 26, 2026 (Australia/Sydney), with the default budgets and
CPU affinity described above. The [48 timing samples](norms-writer-routing-samples.csv)
retain iteration counts, result checks, encoded bytes and shallow configuration
sizes. The [raw process rows](norms-writer-routing-raw.tsv) retain all 168 rows
from the twelve processes, including each memory-scope declaration.

| Case | Rust median ns | Java median ns | Java/Rust | Largest repetition spread |
| --- | ---: | ---: | ---: | ---: |
| Default flush commit | 298,918.945 | 393,630.539 | 1.3168 | 1.0314 |
| Asserted flush commit | 320,214.022 | 369,120.557 | 1.1527 | 1.0336 |
| Default deleted merge + commit | 831,730.539 | 858,655.950 | 1.0324 | 1.0486 |
| Asserted deleted merge + commit | 846,269.294 | 844,186.298 | 0.9975 | 1.0468 |

Each median is the mean of the middle two of six samples. Each case's spread
is the larger of Rust's and Java's maximum/minimum sample ratios. The reporter's
noise floor is the median of those four case spreads: 1.0402. The reciprocal
noise bound is 0.9613. Both flush ratios exceed that floor. Both merge ratios
lie between the reciprocal bound and the floor, so this run cannot resolve a
merge advantage for either engine.

The asserted merge median is about 0.25% slower in Rust. That difference is
inside noise; it is neither evidence of a resolved regression nor proof of a
performance advantage. These whole commit intervals cannot identify an
assertion-loop cause. Java's producer reopening and Rust's retained impact
columns also remain different work within these intervals, as described above.
No optimization was made from an unresolved merge difference.

All 48 result checks match their expected digest, document count, field count
and encoded size. Every timing sample contains at least 1,500 ms of accumulated
operation time (allowing rounding to three decimal places). Each process has
exactly one timing, result check and encoded-size row for each case, and one
shallow configuration row. The independent filesystem readback above used
native Java readers before timing.

| Scoped quantity | Rust | Java |
| --- | ---: | ---: |
| Shallow adapting configuration bytes | 0 | 72 |
| Flush output `.nvm` + `.nvd` bytes | 3,654 | 3,654 |
| Merged output `.nvm` + `.nvd` bytes | 2,909 | 2,909 |

These byte counts are identical in every repetition. The zero-byte Rust
configuration is a static-size result; it does not mean the writer allocates
nothing. Retained buffers, impact columns and the other excluded memory above
remain unmeasured, so full-memory parity is not established.

The full test-framework fixture provenance check reproduces all 50 generators.
The gate's norms replay, scoring and failure tests pass. An unrelated randomized
merge-policy suite exposed an inherited test that Java explicitly disables for
MergeOnFlush's potentially quadratic merges. Matching that suite exclusion leaves
all 43 applicable tests passing with the original failing seed
`18E1E6DA370F454A`; the generic amplification bound remains for other policies.

The full repository gate passes, including release search tests. Workspace
line coverage is 97.65%; every changed production file exceeds the 95% floor.
The new files' line coverage is:

| File | Lines covered |
| --- | ---: |
| `lucene-codecs/src/norms_writer.rs` | 98.08% |
| `lucene-index/src/norms_writer.rs` | 96.13% |
| `lucene-test-framework/src/codecs/asserting_norms_writer.rs` | 99.42% |

The scoped benchmark establishes flush commit performance for this corpus and
leaves merge performance unresolved within observed noise.
