# Installed asserting stored-fields writers

The native `Lucene104` writer can install an operation-local stored-fields
writer through a shared factory. `AssertingStoredFieldsCodec` selects the
existing asserting lifecycle around the production Lucene90 BestSpeed writer.
Ordinary and explicit flushes, append, and merges invoke the installed writer.
The remaining codec graph is unchanged; this is a partial installation of
`AssertingStoredFieldsFormat`, not a complete `AssertingCodec`.

The Java oracle uses actual Lucene 10.5.0 `IndexWriter` and
`AssertingStoredFieldsFormat`. Its direct/imported, sorted/unsorted sessions
verify stored values, live-document order and field-number remapping. Rust
callback counters independently prove installation. Java delegate traces are
retained in the fixture but are not compared with Rust per-overload traces:
Rust merges decode typed values, including binary values, while Java can
forward a binary input stream. Tests also cover ordinary fixed-schema flushes,
pruning before number remapping, reordered/deleted documents, source corruption,
start-versus-decode failures, mapping failures, output closure and
publication/error precedence. Competing-error regressions verify that
`start_document` precedes semantic field decoding, an earlier field-callback
failure precedes later malformed field data, and an earlier missing
field-number mapping retains its typed merge error ahead of later malformed
data. Secondary close errors preserve each original failure.

## Workload and timing boundary

`scripts/bench-stored-fields-writer-routing.sh` pairs
`StoredFieldsWriterRoutingMicro.java` with
`stored_fields_writer_routing_micro.rs`. Each engine prebuilds the same 1,024
rich documents, with an empty stored document every eighth document, six stored
types, 128-byte binary values, an indexed id and numeric rank. Merge preparation
creates eight segments and deletes every seventh document, leaving 877.
Sorted merges order rank ascending. Both engines use in-memory directories,
noncompound output and serial merges without automatic intermediate merges.

Flush timing includes adding all documents, explicit flush and commit. Merge
timing includes force-merge to one segment and commit. Writer construction,
source preparation/deletion commits, decoded-result checking and encoded-size
accounting are outside the measured interval. These are complete writer
operations, including postings and doc values, rather than isolated
stored-fields callbacks. Compare default/asserting cases within each engine
to assess the routing cost. Excluded work can still affect later GC, CPU load
and temperature; repetition spread is reported with the timings.

Java uses a same-wire `Lucene104` FilterCodec alias with an asserting writer
and native readers, avoiding unrelated reader assertions. The fixture uses a
distinct registered codec name; Java's final `Lucene104Codec.storedFieldsFormat`
requires this adapter. No trace recording is timed. Every iteration reads the
actual output and checks its digest against an independent analytic result:
`9753414452126769785` for flush, `5659081544810581733` for unsorted merge and
`5471668284220579839` for sorted merge. The paired reporter requires matching
checks for all six cases in every repetition. Before timing, Rust also writes
an asserted sorted-merged index to disk and default Java Lucene104 readers
verify its digest, 877 documents and one leaf.

## Measurement

Measured October 26, 2026 on the Intel Core i5-13600KF/WSL2 host in the
resource-capped Rust 1.97.1/JDK 21 container. Each run contains three
alternating Rust/Java process pairs pinned to CPUs 0 and 1, with a fixed
128 MiB JVM heap. The 1.5 load guard remains enabled; process boundaries wait
up to 300 seconds for cooldown and then refuse to run if load remains high.
It checks process starts, not load throughout a sample. Warmup/measurement
budgets accumulate timed operation durations, excluding setup and readback.

The first 72 CSV rows (`baseline` and `longer_warmup`) measure the original
implementation, which decoded a whole source document before forwarding its
fields. They predate the error-order repair and remain historical measurements,
not measurements of the final streaming implementation. The final 36 rows
(`corrected_order`) measure the repaired per-field traversal described below.

The first completed baseline used 300 ms warmup and 1,000 ms measurement per
case. Two incomplete launches stopped at the load guard and are excluded.
The resulting baseline medians are nanoseconds per full operation:

| Baseline case | Rust ns | Java ns | Java/Rust | Largest repetition spread |
| --- | ---: | ---: | ---: | ---: |
| Default flush | 1500038.501 | 1489725.768 | 0.99x | 1.04x |
| Asserted flush | 1549983.759 | 1120828.044 | 0.72x | 1.05x |
| Default unsorted merge | 713133.101 | 1142988.999 | 1.60x | 1.02x |
| Asserted unsorted merge | 868372.067 | 980420.663 | 1.13x | 1.05x |
| Default sorted merge | 757084.366 | 914183.329 | 1.21x | 1.01x |
| Asserted sorted merge | 926316.770 | 982142.974 | 1.06x | 1.07x |

The baseline's median repetition noise floor is 1.05x. Default flush is
indistinguishable across engines at that floor. The asserted sorted-merge
ratio also sits inside that case's own 1.07x spread, despite exceeding the
global median floor. Java's later asserted flush runs faster than its default
control, so these fixed-order cases do not isolate assertion overhead or JIT
warmup. Rust asserted/default ratios are 1.033x for flush, 1.218x for unsorted
merge and 1.224x for sorted merge. The complete
[raw timing samples](stored-fields-writer-routing-samples.csv) retain all
repetitions, including the shorter-warmup baseline.

The pre-repair repeat uses 1,000 ms warmup and 1,500 ms measurement with the
same original implementation and workloads:

| Longer-warmup case | Rust ns | Java ns | Java/Rust | Largest repetition spread |
| --- | ---: | ---: | ---: | ---: |
| Default flush | 1510092.482 | 1168338.227 | 0.77x | 1.05x |
| Asserted flush | 1580378.121 | 1076945.762 | 0.68x | 1.07x |
| Default unsorted merge | 713632.941 | 913374.672 | 1.28x | 1.02x |
| Asserted unsorted merge | 919772.591 | 945587.946 | 1.03x | 1.08x |
| Default sorted merge | 762618.112 | 878760.827 | 1.15x | 1.01x |
| Asserted sorted merge | 971272.186 | 942963.366 | 0.97x | 1.06x |

At the repeat's 1.06x median noise floor, both asserted merges are
indistinguishable from Java; neither is a demonstrated speed win. Flush is
slower in both native and asserted configurations. Within-engine ratios of
asserted/default medians are:

| Operation | Rust | Java |
| --- | ---: | ---: |
| Flush | 1.047x | 0.922x |
| Unsorted merge | 1.289x | 1.035x |
| Sorted merge | 1.274x | 1.073x |

These fixed-order controls describe the observed complete operations; Java's
sub-1 flush ratio is not evidence that assertions improve an otherwise equal
fully warmed operation. Longer warmup substantially changed Java's first
default-flush result while Rust's default result remained close. The pair does
not remove all JIT/order/GC effects.

The final run uses the same 1,000 ms warmup and 1,500 ms measurement budgets
after repairing Java's competing-error order. The merge now starts each
output document before decoding its fields and forwards each field immediately.
It no longer constructs a complete source `Document` before writer callbacks.

| Corrected streaming case | Rust ns | Java ns | Java/Rust | Largest repetition spread |
| --- | ---: | ---: | ---: | ---: |
| Default flush | 1465267.576 | 1143799.184 | 0.78x | 1.04x |
| Asserted flush | 1534183.968 | 1063172.233 | 0.69x | 1.05x |
| Default unsorted merge | 714037.162 | 890923.984 | 1.25x | 1.06x |
| Asserted unsorted merge | 851986.716 | 970044.068 | 1.14x | 1.06x |
| Default sorted merge | 761202.246 | 895053.345 | 1.18x | 1.05x |
| Asserted sorted merge | 903283.371 | 973221.526 | 1.08x | 1.07x |

The final run's median repetition noise floor is 1.05x. The asserted unsorted
merge exceeds its own 1.06x spread; the asserted sorted merge exceeds its
1.07x spread by a smaller margin. These are measured wins on these workloads,
with the sorted result close to the observed repetition variation. Both flush
configurations remain slower than Java. The final within-engine controls are:

| Operation | Rust asserted/default | Java asserted/default |
| --- | ---: | ---: |
| Flush | 1.047x | 0.930x |
| Unsorted merge | 1.193x | 1.089x |
| Sorted merge | 1.187x | 1.087x |

All 36 final result checks match across engines and repetitions. The memory
and encoded-size rows are unchanged. The fixed case order still leaves
JIT/order/GC effects unresolved; neither the change between runs nor Java's
sub-1 flush control isolates the cost of a particular callback or allocation.

## Performance disposition

The streaming change repairs Java's callback and error order. It is required
for fidelity, independently of its timing result. No further speculative
optimization was applied. The final below-1 flush result is retained explicitly:
Java/Rust is 0.78x without the optional factory and 0.69x with it. The final
Rust asserted/default median differs by 4.7%. The installed path dispatches
field callbacks, and the existing native lifecycle adapter clones borrowed
values into its current-document buffer before encoding. Default flush encodes
already-owned documents directly. This whole-writer experiment does not
apportion individual allocation/callback costs or identify which other
indexing component accounts for the native flush gap.

Selected merges intentionally decode every surviving document and remap typed
fields through the installed writer. Native default merges can use Doc/Bulk
copy paths, which would skip the assertions being installed. The measured
Rust cost relative to default is therefore retained instead of silently
bypassing callbacks for speed. The repaired visitor removes the source-side
complete-document buffer. Strings and binary values still become owned
per-field temporaries, and the adapter still clones them into its own
current-document buffer. An ownership-aware adapter could remove some copies
in future but would require its own error-timing and lifecycle proof. The
present bounded writer installation makes no claim of general IndexWriter
performance parity or complete AssertingCodec parity.

## Memory scope

Source stored-value payload is 142,390 bytes in each engine, excluding id/rank,
field names, objects and allocation metadata. Incremental writer-state
accounting covers the asserting wrapper only, excluding the native writer,
buffers and directory. Rust reports wrapper size minus native delegate size
plus the boxed sink fat pointer; Java reports the actual wrapper's shallow
size. Both are 24 bytes on the measured host. Rust's stateless factory is zero
bytes; Java's writer-only codec, adapting format and asserting format sum to
72 shallow bytes. These metrics do not measure whole-writer retained memory,
allocation volume, peak memory or RSS.

Default and asserting output sizes agree within each engine. Rust's complete
committed indexes are 40,544 / 35,742 / 36,032 bytes for flush / unsorted merge /
sorted merge. Java's are 40,559 / 35,757 / 36,047 bytes. The 15-byte cross-engine
size difference is retained in the measurements rather than treated as
byte-for-byte index parity; decoded values and shape agree.
