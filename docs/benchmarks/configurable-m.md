# ConfigurableM direct HNSW component

Run `scripts/docker-test.sh bash scripts/bench-configurable-m.sh`. This pairs
Lucene 10.5.0's actual `ConfigurableMCodec.knnVectorsFormat()` with the Rust
`ConfigurableMHnswFormat` adapter. It does not measure Codec/SPI or IndexWriter
installation, streaming callbacks, sorted flush or graph-reuse merges.

The faithful batch adapter passed Java configuration/callback fixtures, exact
four-file differential tests and boundary/error units before measurement. All
sixteen native workloads have Java/Rust median ratios above 1.0. No algorithm
optimization was applied. The M4/648 read advantage is unresolved by its own
case spread; the pair does not establish a statistically reliable advantage for
that case.

## Measured native workloads

Each case uses 500 ms warmup and 750 ms measurement in each of three alternating
engine repetitions through the capped runner, pinned to the same two CPUs.
Java uses assertions, a fixed 128 MiB heap and the vector incubator module;
the runtime confirmed the Panama provider, preferred 256-bit vectors and FMA.
Times are median nanoseconds per complete unit. Ratio is Java/Rust; spread is
the larger engine max/min across repetitions, not a confidence interval.

| M / sparse count | Workload | Rust ns | Java ns | Ratio | Spread |
|---|---|---:|---:|---:|---:|
| 128 / 647 | Read | 148,869.815 | 216,848.769 | 1.46x | 1.10x |
| 128 / 647 | Unfiltered search | 61,467.301 | 140,995.638 | 2.29x | 1.03x |
| 128 / 647 | Filtered search | 65,022.129 | 124,636.557 | 1.92x | 1.07x |
| 128 / 647 | Build/write | 26,878,041.893 | 38,343,983.550 | 1.43x | 1.09x |
| 128 / 648 | Read | 198,415.612 | 247,976.747 | 1.25x | 1.07x |
| 128 / 648 | Unfiltered search | 33,657.050 | 64,925.324 | 1.93x | 1.12x |
| 128 / 648 | Filtered search | 47,551.950 | 71,811.973 | 1.51x | 1.06x |
| 128 / 648 | Build/write | 39,305,542.750 | 55,188,401.286 | 1.40x | 1.07x |
| 4 / 647 | Read | 187,744.953 | 234,732.720 | 1.25x | 1.11x |
| 4 / 647 | Unfiltered search | 57,913.672 | 129,074.891 | 2.23x | 1.03x |
| 4 / 647 | Filtered search | 60,484.724 | 122,479.852 | 2.02x | 1.15x |
| 4 / 647 | Build/write | 36,750,999.000 | 52,583,502.600 | 1.43x | 1.07x |
| 4 / 648 | Read | 294,208.620 | 322,131.433 | 1.09x | 1.12x |
| 4 / 648 | Unfiltered search | 25,423.673 | 50,567.798 | 1.99x | 1.02x |
| 4 / 648 | Filtered search | 39,360.500 | 55,980.060 | 1.42x | 1.05x |
| 4 / 648 | Build/write | 53,277,002.933 | 73,171,527.818 | 1.37x | 1.04x |

The report's aggregate floor is 1.07x, which does not override a larger
individual case spread. Short 20/30 ms smoke timings are not these results.

## Result checks and timing boundaries

Each of the four segments has six fields: dense float/byte vectors with
1,294/1,296 documents, sparse float/byte vectors with 647/648 values, and empty
float/byte fields. Sparse 647 has no graph and sparse 648 has a graph under the
actual selected format's threshold 100. Dense fields have graphs in both cases.
All fields use dimension four, Euclidean similarity, insertion order and seed
42. The Java global builder seed is restored in `finally`.

Before timing each segment, both writers must reproduce all four actual fixture
files exactly. Read units open a reader, check integrity, consume every vector
value/doc mapping and every graph level/node/neighbor, then close. Rust checks
data CRCs on open as well as during integrity; Java defers data CRCs until the
integrity call. Rust returns owned vector/neighbor buffers, whereas Java reuses
scratch and graph cursors. These API costs are included.

Search uses preopened readers, a precomputed `doc % 3 != 0` mask, k=7 and
`Integer.MAX_VALUE` visits. Both engines allocate collectors and query vectors
inside each unit, run the same three queries on all six fields and consume
ordered docs, raw score bits, visits and termination relation. Filtered and
unfiltered units are separate.

Build/write uses precomputed identical source vectors. Timing includes a fresh
owned Rust input clone versus Java `addValue` buffering, graph construction,
encoding and consumption of all four files. Rust replays flat ordinals at flush;
Java starts replay when the threshold is crossed during `addValue`. Rust returns
owned buffers; Java uses `ByteBuffersDirectory` and copies output into arrays.
The pair includes those ownership/publication costs and does not isolate graph
construction from encoding. Every timed/warmup batch checks its digest; all
seventeen checks, including the construction observation below, are mandatory
across all six engine outputs.

## Construction observation and memory limits

The separate construction case creates M128/default, M4, M1 and M512 codecs and
checks names, accessors and stable selected-format identity. It observed 11.576
ns Rust and 670.858 ns Java per four-codec unit. Java constructs its default
Lucene104 component graph; Rust's default delegate is a routing descriptor.
These are different assembly scopes. Their timing ratio is not evidence of
equivalent whole-codec construction performance and is excluded from the native
workload table and conclusions.

Memory metadata reports serialized file payload lengths only; exact byte
equality establishes equal serialized size. It excludes capacities, native
graphs/readers/writers, temporary buffers, allocator/runtime costs and RSS.
No peak, whole-component or graph-memory parity is established by this pair.

Correctness evidence is `GenConfigurableM`, `configurable_m_fixtures.rs` and
module units, including every native value/arc/search, wrong segment ID,
checksum/truncation/suffix rejection, configured M1/512 empty/singleton metadata,
delegate identity/errors and invalid input before publication. The partial API
also omits inherited displays and the format's fixed max-dimensions accessor.
