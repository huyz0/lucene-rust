# BulkScorerWrapperScorer traversal

Run `scripts/docker-test.sh bash scripts/bench-bulk-scorer-wrapper.sh`.
Three alternating Rust/Java pairs use the same two CPUs, 500 ms warmup and
750 ms measurement per case. Java runs Lucene 10.5.0's actual wrapper with
assertions, the vector incubator module and a fixed 128 MiB heap.

Each batch wraps the same 8,192 sorted documents and raw score bits, using
buffer size 64. Dense documents have gap 1; sparse documents have gap 17.
`next` consumes every document; `advance` seeks the current document plus 17.
The deterministic bulk sources collect `[min,max)` and return the window
maximum as a legal lower bound, so sparse cases include empty refill windows.

Wrapper/source/iterator construction, initial buffer allocations, result checks and teardown
are outside the measured interval. Refills, collection, cursor movement,
binary search and the document/score checksum are timed. Java's refill helper
construction remains timed; the JIT may eliminate those allocations. Every batch checks
the checksum and hit count, and the report requires matching checks in every
engine repetition. Java may reclaim untimed setup allocations in a later
timed interval; its GC lifecycle and Rust's deterministic destruction remain
part of this observation.

## Faithful baseline

Times are median nanoseconds per traversal batch; ratios are Java/Rust.
Noise is the larger same-engine max/min spread across repetitions.

| Case | Rust ns | Java ns | Ratio | Noise |
| --- | ---: | ---: | ---: | ---: |
| Dense next | 58,589.543 | 51,599.073 | 0.88 | 1.18 |
| Dense advance | 8,218.341 | 8,683.410 | 1.06 | 1.14 |
| Sparse next | 41,178.792 | 42,575.692 | 1.03 | 1.12 |
| Sparse advance | 42,874.636 | 36,505.681 | 0.85 | 1.46 |

The faithful port repeats a binary search even when the target is the first
remaining buffered hit. This work appears on every sequential hit and most
sparse advances. The baseline is noisy, particularly sparse advance, so the
numbers alone do not establish a stable regression for every sub-1.0 case.

## Selected optimization

The wrapper checks the first remaining buffered hit before entering the
original binary-search fallback. Strictly increasing document IDs make that
hit the same insertion point the search would return. Refills, two separate
arrays, source errors, partial-collection failure state and cursor semantics
are retained. No allocation or retained state was added.

| Case | Rust ns | Java ns | Ratio | Noise |
| --- | ---: | ---: | ---: | ---: |
| Dense next | 25,925.628 | 49,178.471 | 1.90 | 1.01 |
| Dense advance | 7,606.312 | 8,548.782 | 1.12 | 1.18 |
| Sparse next | 29,910.967 | 42,862.350 | 1.43 | 1.10 |
| Sparse advance | 28,443.773 | 36,250.123 | 1.27 | 1.01 |

All selected medians meet 1.0. Dense advance's apparent advantage is within
the observed repetition spread; it does not establish a reliable speedup.
The two 64-entry buffers retain the same 512 logical payload bytes on each
side. This excludes object headers, references, padding, allocator metadata,
source/input arrays and total process memory.

## Correctness and remaining scope

Java differential fixtures exercise 152 synthetic operations across buffer
sizes 1, 2, 4, 7 and 128, exact score bits (including signed zero, infinities
and a NaN payload), sparse empty windows, advance/binary-search paths,
exhaustion, shared iterator state, original I/O-error identity and retries
after both empty and partial failed collections. Sixteen further cases use
real indexed MatchAllDocs and constant-score term bulk scorers in both
engines. Unit tests cover explicit invalid-buffer and overflowing-window
failures and retained source errors. These tests pass before and after the
optimization; targeted clippy passed on the final implementation and example.

The generic source forwards `cost()` exactly. `LeafBulkSource` owns the actual
Rust indexed bulk scorer and takes a known cost explicitly, because the
optimized `LeafBulkScorer` variants have no common intrinsic cost accessor.
The fixture uses independently known costs of 97 and 9; no second scorer is
created to estimate them. Intrinsic arbitrary-query cost remains a gap.
Rust rejects zero/negative buffer sizes and returns an error for overflowing
refill windows; Java can loop forever with a zero buffer and wraps window
arithmetic. Rust represents shared iterator state through successive
exclusive borrows. Indexed-query throughput, intrinsic arbitrary-query cost
and total retained memory are not measured by this traversal benchmark.
