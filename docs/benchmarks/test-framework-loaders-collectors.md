# Test-framework resource loader and hit collector

These standalone benchmark pairs compare the Rust test-framework helpers
with Lucene 10.5.0 in the capped test container (Rust 1.97.1, JDK 21).
Rust runs as examples of `lucene-test-framework`, keeping that crate out
of production dependencies.

## StringMockResourceLoader

`StringMockResourceLoaderMicro.java` and `string_resource_loader_micro.rs`
open and consume resources containing 8 or 256 repetitions of
`"café 😀\0\n"`: 96 or 3,072 UTF-8 bytes. Each batch opens 256 resources and
sums every byte. All result checks matched across engines and repetitions.
The harness requires both result checks in every engine and repetition;
the gate tests missing checks, duplicate rows and mismatched results against
the actual shared report parser.

Three alternating repetitions ran pinned to CPUs 0 and 1. Each case used
1,000 ms warmup and 1,500 ms measurement; the JVM heap was fixed at 128 MiB.

| Resource bytes | Rust ns/open-and-consume | Java ns/open-and-consume | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| 96 | 18.717 | 97.374 | 5.20x | 1.05x |
| 3,072 | 189.590 | 3,024.796 | 15.95x | 1.03x |

Both timing differences exceed their measured noise. These are complete
resource API operations: Java transcodes its string to UTF-8 and consumes
a `ByteArrayInputStream` through `read()`, while Rust copies UTF-8 bytes and
iterates the returned `Vec`. The comparison does not isolate allocation
or transcoding speed. The report presents memory rows separately and excludes
them from timing statistics.

| Resource bytes | Rust retained bytes | Java retained bytes | Rust opened-resource bytes | Java opened-resource bytes |
|---|---:|---:|---:|---:|
| 96 | 120 | 200 | 120 | 144 |
| 3,072 | 3,096 | 4,664 | 3,096 | 3,120 |

Memory figures estimate owned storage. Rust counts the struct and backing
capacity; Java counts shallow objects and backing arrays using
`RamUsageEstimator`. Every input contains supplementary Unicode, forcing
Java's string backing storage to UTF-16. These figures exclude allocator
metadata and runtime memory; they are not process RSS.

```sh
scripts/docker-test.sh bash scripts/bench-string-resource-loader.sh
```

## DummyTotalHitCountCollector

`DummyTotalHitCountMicro.java` and `dummy_hit_bench.rs` replay the same
4,096 xorshift-generated document IDs. Each batch creates two collectors,
selects one by document-ID parity, and checks that their counts total
4,096. The count-derived digest was 65,386 in every repetition.

Three alternating repetitions ran pinned to CPUs 0 and 1, using 1,500 ms
warmup and 3,000 ms measurement.

| Case | Median Rust ns/callback | Median Java ns/callback | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Two collector counters | 0.680 | 0.858 | 1.262x | 1.043x |

The timing difference exceeds the measured noise. This measures callback
counting, including collector setup and validation amortized over each
batch. It does not measure query execution, manager reduction or parallel
collection. Neither helper required algorithm optimisation.

Peak process RSS ranged from 11,824 to 11,964 KiB for Rust and from
105,496 to 109,116 KiB for Java; medians were 11,940 and 105,908 KiB.
RSS includes runtime startup, the JVM heap and GC, and benchmark input.
The two counters carry eight bytes of integer payload in both
implementations; Java object and leaf headers are additional.

The result parser requires one matching, nonempty result per engine.
Adversarial checks confirmed rejection of empty results, duplicate rows,
digest mismatches, a missing engine, zero operations, nonnumeric timing,
count and digest fields, and NaN timing. These checks
validate report structure and agreement; they cannot prove that the twins
implement the intended algorithm. Java differential tests separately
compare real searches over one and three segments, deletions and missing
queries.

```sh
scripts/docker-test.sh bash scripts/bench-dummy-hit-collector.sh 1500 3000
```
