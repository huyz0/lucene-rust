# XY shape test helpers

`shape_test_util_micro.rs` and `ShapeTestUtilMicro.java` measure Lucene 10.5.0's
actual `ShapeTestUtil` methods and the faithful Rust port. Java explicitly
installs `java.util.Random` in randomizedtesting's context; context construction
is outside timing and each actual context lookup remains timed. Nightly is
false. Each batch resets its RNG and consumes every output coordinate's raw
float bits or every oracle result, plus RNG continuation. All five checked
digests match in every engine and repetition.

The capped Rust 1.97.1/JDK 21 container ran three alternating paired repetitions
on CPUs 0 and 1, with 1,000 ms warmup, 1,500 ms measurement per case and a fixed
128 MiB JVM heap. Medians are nanoseconds per complete batch.

| Batch | Rust ns | Java ns | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| 512 biased floats | 3,573.394 | 11,179.196 | 3.13x | 1.01x |
| 32 boxes, points and circles | 2,329.970 | 11,404.573 | 4.89x | 1.01x |
| Eight polygon/line pairs | 7,737.094 | 13,804.463 | 1.78x | 1.01x |
| Sixteen regular 64-gons | 21,873.700 | 25,987.929 | 1.19x | 1.01x |
| 1,024 slow-containment calls | 23,112.242 | 31,391.596 | 1.36x | 1.01x |

Every measured difference exceeds the noise observed in this run. The faithful
port needed no algorithm optimisation. Rust borrows geometry coordinate arrays
where Java's getters clone float arrays; both slow oracles materialize double
arrays. Primitive benchmark consumption uses stack arrays in Rust and temporary
arrays in Java. These are complete helper workloads including the respective
APIs and allocations.

One RNG graph occupies 24 bytes in Rust and 56 bytes in Java, counting Java's
`Random` and its `AtomicLong` once. One retained five-vertex polygon occupies
136 bytes in Rust and 144 bytes in Java, counting its object/struct, coordinate
arrays and holes vector/array. These counts exclude temporary arrays, unrelated
shapes, allocator metadata, runtime and RSS; they make no peak-memory claim.

```sh
scripts/docker-test.sh bash scripts/bench-shape-test-util.sh 1000 1500
```
