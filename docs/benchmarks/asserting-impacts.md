# Asserting reader impacts

The pair runs the actual Lucene 10.5.0
`AssertingLeafReader.AssertingImpactsEnum` and Rust's `AssertingImpactsEnum`
over shared synthetic documents, frequencies, impact lists and explicit
operation traces. Python generates an independent expected-value digest;
every iteration must match it, and every engine/repetition must agree.

The dense case walks 4,096 documents with one impact level/pair. The multilevel
case advances through 4,096 gap-three documents with three levels of 32 pairs.
The positions case mixes next/advance over 2,048 gap-four documents, two levels
of eight pairs, frequencies one through four, and all positions, offsets and
payloads. Every synthetic document has norm one. Each level's first pair is
`(4, 1)`, so it bounds every document; remaining pairs increase in frequency
and unsigned norm, crossing the signed norm boundary in the larger lists.
Higher levels have identical lists and larger fixed document bounds.

Timing includes full trace dispatch, actual wrapper checks, delegate traversal,
impact materialization and validation, immediate view consumption and digest
calculation. Input parsing, shared data construction, delegate/wrapper setup,
final digest comparison and wrapper destruction are outside timing. Both
engines consume document IDs, frequencies, positions, offsets, payload bytes,
every impact bound/pair and terminal state. Java assertions are enabled; Rust
assertions remain enabled in release.

Rust returns owned cloned impact vectors; Java returns borrowed buffers and
an asserting view, with scratch copies during validation. Those costs are
included. Shallow wrapper sizes count Rust's inline postings checker and Java's
separately owned postings checker. They exclude delegates, shared input/trace,
allocator metadata and transient impacts/views/validation copies. Peak
allocation, GC and RSS are unmeasured. Shallow sizes cannot establish whole
memory parity. Fixed synthetic blocks measure neither codec I/O nor search,
and immediate consumption does not exercise retained-view lifetimes.

```sh
scripts/docker-test.sh bash scripts/bench-asserting-impacts.sh 1000 1500
```

The script alternates engines across three repetitions, pins both to CPUs
zero and one, uses a 128-MiB Java heap, and rejects load averages above 1.5
before building and before each engine invocation.

## Measurements

Measured October 26, 2026 on an Intel Core i5-13600KF under WSL2
6.18.54.1, with Rust 1.97.1 (`8bab26f4f`) and Temurin
21.0.12.1+1-LTS. Values are nanoseconds per complete trace, with medians
from three alternating process pairs. A ratio above one favors Rust.

| Case | Rust median | Java median | Java/Rust |
| --- | ---: | ---: | ---: |
| Single | 189210.252 | 83168.070 | 0.44x |
| Multilevel | 1181181.819 | 1291302.110 | 1.09x |
| Positions | 212926.134 | 223947.259 | 1.05x |

Raw repetitions, in process order within each engine:

| Case | Rust | Java |
| --- | --- | --- |
| Single | 189278.980, 188890.861, 189210.252 | 82833.773, 83295.514, 83168.070 |
| Multilevel | 1179764.668, 1181594.111, 1181181.819 | 1250639.704, 1291302.110, 1293547.117 |
| Positions | 218375.165, 212926.134, 212317.832 | 223151.005, 224738.674, 223947.259 |

The largest within-engine spread is about 1.03x. The positions result is
close to that noise floor; it does not establish a broad speed advantage.
Every iteration and all six processes per case matched the independent
digest: single `17142866889888480579` (21,508 operations), multilevel
`13919717366214806851` (21,508 operations), positions
`10868585420572058435` (26,116 operations).

Shallow wrapper storage is 40 bytes Rust and 64 bytes Java with the scope
defined above. This does not establish peak or retained whole-graph memory
parity.

## Unwrapped control

```sh
scripts/docker-test.sh bash scripts/bench-asserting-impacts.sh 1000 1500 raw
```

This runs the single-pair input in fresh processes with the same trace,
delegate, expected digest and timing boundaries, without either asserting
wrapper. Rust still clones the owned nested vectors required by the existing
`ImpactsEnum::impacts` contract; Java still returns its preallocated borrowed
buffers. Removing wrappers also changes dispatch and compiler optimization
opportunities. The comparison measures aggregate traversal costs, and does
not isolate allocation, dispatch or individual assertion costs.

| Control | Rust median ns/trace | Java median ns/trace | Java/Rust |
| --- | ---: | ---: | ---: |
| Single, no wrapper | 110656.847 | 46462.470 | 0.42x |

Rust repetitions were 109881.291, 110656.847 and 110870.684 ns; Java
repetitions were 46431.007, 46462.470 and 48545.947 ns. The largest spread
was 1.05x. Every iteration and all six processes matched the single-case
digest, operation count and document count above.

## Below parity and why it is left

The single-pair deficit persists without the asserting wrapper: 0.42x raw
versus 0.44x wrapped. The new wrapper therefore does not explain the entire
cross-engine deficit. A source-visible difference remains in both workloads:
the Rust delegate clones an outer vector and each level's vector on every
impact retrieval, whereas the Java delegate returns existing buffers. This
comes from the existing reader trait, not a new cache or representation in
this port. Trace dispatch and JIT/AOT optimization also differ. No allocation
profile was taken, so these controls do not prove that ownership alone causes
the deficit or assign it a precise cost.

The faithful asserting wrapper retains this measured single-case exception.
Replacing the shared reader trait with borrowed impacts would affect readers,
callers and retained-view semantics beyond this wrapper; that redesign needs
its own profile and correctness scope. No speed optimization was applied to
the validated Java algorithm. The multilevel and positions measurements do
not remove the single-case performance debt, and the shallow storage result
does not establish whole-memory parity.
