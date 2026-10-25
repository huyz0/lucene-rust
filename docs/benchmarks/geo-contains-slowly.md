# Geographic slow polygon containment

`GeoTestUtil.containsSlowly` is a test oracle over latitude/longitude arrays,
using planar ray crossing with inclusive boundaries. The Rust port preserves
Java's hole rejection before the bounding-box check, edge traversal, boolean
conditions and floating-point operation order. It adds no query-coordinate
validation or dateline normalization. Holes return a typed error with Java's
exact message.

The faithful baseline copied both coordinate arrays on every call, as Java's
getters do. It passed 8,009 actual-Java comparisons across 16 polygons before
measurement. The small-polygon baseline was slower than Java. The measured
optimization replaces just those two coordinate copies with immutable borrowed
slices. The same differential and unit tests pass afterward.

## Workloads and measurement

`GeoContainsSlowlyMicro.java` calls the actual Lucene 10.5.0 helper;
`geo_contains_slowly_micro.rs` calls the Rust helper. An integer-coordinate
L-shaped polygon has seven vertices, or 1,025 when every edge is subdivided
into unit segments. The inputs use identical representable coordinates and
128 queries per batch. Their independent analytic membership formula is
`bbox && (latitude <= -16 || longitude <= -48)`.

Full-scan queries avoid edges and mix 64 interior with 64 concavity-exterior
points inside the bounding box. Rejection queries cover all four bounding-box
bounds. Boundary queries lie strictly within the first horizontal segment.
Polygon/query construction and expected digests are outside timing. Timed work
includes helper calls, digest accumulation, per-batch digest validation,
result consumption and loop timing. Rust black-boxes the polygon and query
slice; Java publishes the digest to a volatile sink. Every process checks all
results against the analytic digest, and the paired reporter requires matching
digests, query counts and vertex counts in every repetition.

Measured October 26, 2026, using the standard capped Rust 1.97.1/JDK 21
container on the Intel Core i5-13600KF/WSL2 host, pinned to CPUs 0 and 1.
Each phase has three alternating Rust/Java process pairs, with 1,000 ms warmup
and 1,500 ms measurement per case, and a fixed 128 MiB JVM heap. The load guard
remains enabled. Two baseline launches stopped at that guard after compilation;
they produced no timing samples and are excluded. Medians below are nanoseconds
per 128-query batch. [All samples](geo-contains-slowly-samples.csv) are retained.

| Copied-coordinate baseline | Rust ns | Java ns | Java/Rust | Largest repetition spread |
| --- | ---: | ---: | ---: | ---: |
| Small full scan | 3064.223 | 2950.843 | 0.96x | 1.02x |
| Large full scan | 150621.437 | 274723.067 | 1.82x | 1.01x |
| Large bbox rejection | 13488.443 | 125004.060 | 9.27x | 1.03x |
| Large boundary exit | 13962.483 | 124403.029 | 8.91x | 1.01x |

| Borrowed-coordinate implementation | Rust ns | Java ns | Java/Rust | Largest repetition spread |
| --- | ---: | ---: | ---: | ---: |
| Small full scan | 1279.365 | 2959.006 | 2.31x | 1.02x |
| Large full scan | 136683.074 | 275283.729 | 2.01x | 1.01x |
| Large bbox rejection | 201.842 | 124483.132 | 616.74x | 1.04x |
| Large boundary exit | 409.202 | 126616.314 | 309.42x | 1.03x |

All final differences exceed the observed repetition spread. The large ratios
for rejection and early exit compare an allocation-free borrowed-slice helper
with Java's actual helper, which clones coordinates before either return.
They do not describe indexed queries, spherical geometry or general search
performance. The controlled source change measures removing allocation,
copying and destruction together; it does not isolate allocator or copy costs.

## Memory scope

Retained polygon estimates are equal: 224 bytes for the small ring and
16,512 bytes for the large ring. Rust counts its inline `Polygon` and coordinate
payloads; Java counts the shallow polygon, both coordinate arrays and empty
holes array using `RamUsageEstimator`. Shared enums and benchmark query arrays
are excluded, as are allocator metadata, runtime and RSS.

The copied Rust baseline materializes 112 or 16,400 coordinate payload bytes
per call; the final Rust helper borrows them and materializes none. Java's
source-level clone estimates have the same coordinate payloads. With array
headers and the empty holes-array clone, the totals are 160 or 16,448 bytes.
These are source-derived estimates before possible JIT allocation elimination,
not measured allocation rates, GC costs or peak-memory claims.

```sh
scripts/docker-test.sh bash scripts/bench-geo-contains-slowly.sh 1000 1500
```

To reproduce the baseline, change the helper's `poly_lats()` and `poly_lons()`
borrows to `.to_vec()` copies and set the example's
`COORDINATE_COPIES_PER_CALL` to 2. No algorithm, input, release profile or
measurement setting changes between the phases.
