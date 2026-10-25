# BaseXY point generators and oracles

Run `scripts/docker-test.sh bash scripts/bench-base-xy-point.sh`. The pair uses
Lucene 10.5.0's actual inherited `BaseXYPointTestCase` methods and the Rust
`geo::base_xy_point` module. This measures the bounded generator/hook/oracle
surface, not indexing or query verification.

The faithful implementation passed Java differential fixtures and boundary/error
units before measurement. All eight checked workloads are faster than Java in
this pair, so no algorithm optimization was applied.

## Measured pair

Each case has 500 ms warmup and 1,000 ms measurement in each of three alternating
Rust/Java repetitions, pinned to the same two CPUs through the capped runner.
Java uses assertions, the vector incubator module, a fixed 128 MiB heap,
`tests.nightly=false` and `tests.multiplier=1`. Times are median nanoseconds per
complete batch; the ratio is Java/Rust. Spread is the larger engine max/min
across the three repetitions; it is not a confidence interval.

| Workload | Rust ns | Java ns | Ratio | Spread |
|---|---:|---:|---:|---:|
| Random tiny | 290.371 | 3,020.025 | 10.40x | 1.09x |
| Random medium | 21,552.275 | 242,221.401 | 11.24x | 1.07x |
| Same point | 5,118.071 | 6,224.050 | 1.22x | 1.03x |
| Low cardinality | 7,057.284 | 76,190.622 | 10.80x | 1.04x |
| All X equal | 13,923.332 | 164,785.429 | 11.84x | 1.06x |
| All Y equal | 13,828.462 | 162,183.871 | 11.73x | 1.03x |
| Default shape hooks | 12,061.609 | 30,066.100 | 2.49x | 1.07x |
| Rectangle/distance oracles | 4,399.074 | 9,758.925 | 2.22x | 1.05x |

The aggregate report floor is 1.05x. Every case exceeds both that floor and its
own spread. Short 20/30 ms smoke measurements validated the harness and are not
the reported performance result.

## Equivalent results and scope

Every batch resets seed 42. Each generator invokes the same four capture hooks;
their checksum consumes callback order, both array lengths, every raw float bit
and RNG continuation. The all-Y copy branch retains Java's assignment behavior.
The hooks case consumes sixteen coordinate/box/polygon/geometry sequences with
raw bits and geometry types. The oracle case consumes 1,024 rectangle/distance
pairs, including raw double result bits. All eight result checks are mandatory
in all six engine outputs, and each warmup/timed batch verifies its digest.

Java randomized context, RNG, capture and oracle construction are outside timing.
Actual inherited context lookups, seed reset, arrays, shape creation and capture
walks remain timed. A shared-stream check outside timing verifies that the
randomizedtesting wrapper forwards to the supplied `Random`; object identity is
not required. Java polygon getters clone arrays and checksum helpers create small
float arrays, whereas Rust borrows coordinate slices and uses stack arrays.
These API/ownership costs are included in the hooks result. The ratio does not
isolate RNG arithmetic or shape-generation cost from result consumption.

## Memory scope

| Selected owned state | Rust bytes | Java bytes |
|---|---:|---:|
| Capture shallow state | 8 | 40 |
| Two representative 1,000-float arrays, including headers | 8,048 | 8,032 |
| Five-vertex polygon and coordinate arrays | 136 | 144 |

Rust array headers cost 16 bytes more in this representative case; its capture
state costs 32 bytes less. These rows exclude allocator metadata, transient
copies, unrelated geometry, randomized context, runtime and RSS. They do not
establish peak or whole-suite memory use.

Correctness evidence is `GenBaseXYPoint`, `base_xy_point_fixtures.rs` and module
units: 240 generator runs, 3,200 default-hook calls, 768 rectangle cases and 169
distance cases, including both nightly modes and signed-long seed extremes.
The full query verifiers, fixed/multivalued suites, contextual codec setup and
query equality/hash checks remain unported.
