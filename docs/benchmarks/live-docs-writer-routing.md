# Installed asserting live-docs writer

`AssertingLiveDocsCodec` installs the existing `AssertingLiveDocsFormat` around
native Lucene90 live-docs output. The configured `IndexWriter` uses it for
term/query deletes, private deletes at flush, reader-ID deletes, filtered
imports and deletes carried while a merge runs. The index retains generation
advancement, sync and publication. An operation-local directory tracks created
outputs so returned failures attempt to reclaim only those outputs while
preserving the primary error and preexisting files. Cleanup failures can leave
an abandoned file; the retry uses a later generation.

This is a writer-only installation. Native readers and the `Lucene104` codec
name remain fixed. Whole AssertingCodec composition, reader routing and
IOContext remain partial. Standalone delete helpers and ConcurrentIndexWriter's
fresh buffered-delete path retain native output. Recovery covers returned I/O
errors: callback panics unwind without guaranteed output cleanup or advancement
of the retry generation.

## Correctness evidence

`GenLiveDocsWriterRouting` runs actual Lucene 10.5.0 `IndexWriter` sessions with
a registered FilterCodec whose live-docs writer enters the actual assertion
wrapper and native delegate. The fixture records sorted and unsorted committed
masks, deletion generations/counts, physical stored IDs and delegate state
across term/query/noop deletes, reopen, append and merge.
`live_docs_writer_routing.rs` compares complete snapshots and delegate-state
multisets. Only callback ordering and segment names are excluded from the
multiset; persisted masks, counts, generations and physical IDs remain exact.

Seven integration regressions additionally cover configured writer reset,
incremental counts, private deletes before first flush, filtered imports,
deletes carried during a bounded paused merge, errors before creating output,
errors after native close, sync failures, competing cleanup failures, and
preexisting candidate files. Failed attempts preserve the committed view and
skip the failed generation on retry. A temporary unconditional deletion of
the candidate output made the preexisting-file regression fail; after restoring
the source with a fresh modification timestamp and rebuilding, it passed.
Adapter units compare exact native bytes, generation 36's base-36 suffix,
length/count/overflow assertions before creation and typed delegate I/O errors.

## Workload and timing boundary

`scripts/bench-live-docs-writer-routing.sh` pairs
`LiveDocsWriterRoutingMicro.java` and `live_docs_writer_routing_micro.rs`.
Each iteration creates the same 1,024 indexed ID documents in eight segments
of 128 documents. First deletion commits remove every seventh document,
leaving 877. Subsequent deletion commits begin with that first generation
already committed, then remove multiples of eleven that remain live, leaving
797. Both engines use in-memory directories, noncompound output, a merge
factor of 1,000, a 128-document flush limit and disabled RAM auto-flushing.
No merge occurs in these cases.

Only `commit` is timed. Corpus construction, flushes, prior commits, deletion
queuing, readback and teardown are outside each interval. A guard rejects any
candidate generation written before the timer starts. Every measured iteration
checks all live bits, an analytic digest, document counts, eight leaves and
eight current-generation `.liv` files. The paired report requires matching
digest/count/leaf-count/encoded-byte checks for all four cases in every
repetition. Encoded byte totals measure the current `.liv` generations only,
not complete index sizes or byte-for-byte cross-engine identity.

Before timing, the asserted Rust writer also writes a filesystem index that
default Java Lucene104 readers open: 797 live documents across eight leaves,
digest `4339461892599487424`, and 600 bytes of current live-docs output.

## Measurement scope

The pair uses the capped Rust 1.97.1/JDK 21 container, CPUs 0 and 1 and a fixed
128 MiB JVM heap. Three repetitions alternate Rust/Java, Java/Rust and
Rust/Java. Each case accumulates 1,000 ms of warmup and 1,500 ms of measured
commit time. The load guard checks process starts, waits up to 300 seconds
for load average to fall to 1.5, then refuses to proceed. It does not monitor
load throughout a sample. Excluded preparation can still affect GC, load and
temperature; wall time is longer than the accumulated timing budgets.

The shallow codec configuration values exclude directory state, created-file
tracking sets, bitset/encoder buffers, the native writer graph, allocator and
runtime overhead, and RSS. They do not establish whole-writer memory parity.

Java runs without `--add-modules jdk.incubator.vector` and reports that its
vector provider falls back. Results describe this configuration; they do not
establish performance against Vector API-enabled Lucene.

## Results

Measured October 26, 2026. The [24 timing samples](live-docs-writer-routing-samples.csv)
retain all three repetitions for each engine and case. Values below are
median nanoseconds per deletion commit; Java/Rust above 1.0 favors Rust.

| Case | Rust ns | Java ns | Java/Rust | Largest repetition spread |
| --- | ---: | ---: | ---: | ---: |
| Default first commit | 241594.522 | 256487.193 | 1.06x | 1.04x |
| Asserted first commit | 243424.245 | 249124.072 | 1.02x | 1.03x |
| Default subsequent commit | 161897.865 | 165589.167 | 1.02x | 1.01x |
| Asserted subsequent commit | 162828.069 | 168096.848 | 1.03x | 1.03x |

The report's median repetition noise floor is 1.03x. Both asserted cases
are within their own largest repetition spread: the subsequent ratio is
1.03236x against a 1.03330x spread. These runs do not resolve a speed
advantage for the asserted Rust writer. The default first-commit control has
a small measured advantage; the default subsequent result lies within the
report's global noise floor. These are whole commit measurements, including
delete resolution, metadata and directory work, rather than isolated live-docs
encoding or assertion costs.

Rust asserted/default median time ratios are 1.008x for first commits and
1.006x for subsequent commits. Those deltas are smaller than the measured
noise and do not isolate wrapper overhead. Java's asserted first case also
runs faster than its default control; fixed within-process case order and JIT
effects prevent treating that contrast as an assertion-speed measurement.
The faithful implementation's medians are not slower than this Java pair;
these results do not justify a further routing redesign. The existing
asserting-format popcount checks and native encoder remain unchanged.

Every case and repetition produced 600 current-generation live-docs bytes per
engine. Shallow adapting codec/format configuration is 0 bytes in Rust and
72 bytes in Java, under the exclusions above. Runtime tracking allocations
and whole-writer memory retention remain unmeasured. No general IndexWriter
throughput or total-memory parity claim follows from this bounded pair.
