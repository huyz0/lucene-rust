# Integer FST enumeration

`micro_ints_ref_fst_enum.rs` and `IntsRefFstEnumMicro.java` enumerate and
seek the same Lucene 10.5.0 BYTE4 FST, using `PositiveIntOutputs`. The
committed continuous-node fixture contains 596 keys, including the empty
input, supplementary code points and a 30-label key. Setup and trace
parsing are outside timing. Every timed batch creates a fresh enumerator.

`next` walks all keys and hashes every label and output. Each seek batch
uses the keys and each key with an appended label 1, in reverse order,
checking prefix reuse and rewinds. Exact, ceiling and floor results hash
the accepted labels and output, or a missing-result marker.

Three interleaved Rust/Java pairs used 3,000 ms warmup and 3,000 ms
measurement per operation, pinned to CPUs 0 and 1 in the capped container
(Rust 1.97.1, JDK 21). Each engine runs the four operations in one process.

| Operation | Rust ns/key or probe | Java ns/key or probe | Java/Rust | Same-engine noise |
|---|---:|---:|---:|---:|
| Ceiling | 49.542 | 49.066 | 0.99x | 1.03x |
| Exact | 26.821 | 30.835 | 1.15x | 1.19x |
| Floor | 29.375 | 29.930 | 1.02x | 1.05x |
| Next | 19.893 | 19.807 | 1.00x | 1.09x |

Median ratio is 1.01x; the shared report's noise floor is 1.07x. Ceiling,
floor and next are indistinguishable from parity under that rule. The
exact-seek median exceeds the shared floor, but its own repetitions vary
by 1.19x, so this run does not establish a stable 15% improvement. The
ceiling and next medians slightly below 1.0 are left because neither is a
resolved slowdown. This result covers this fixture and workload, not all
FST sizes or output types.

The faithful baseline initialized arc/output Vecs at capacity one;
`next` measured 20.367 ns/key against Java's 18.235 ns/key (0.90x).
An allocation profile with `libmemusage.so` showed ten reallocations per
walk: four each for arcs and outputs, two for labels. Reserving ten slots
for arcs and outputs matches Java's initial array capacity. The rebuilt
profile recorded 992,924 reallocations with approximately 165,384 walks,
consistent with six per walk. These are allocation counts, not CPU
attribution. The two timing runs also differ in Java time and noise, so
the change cannot be credited with closing the entire baseline gap.

Reservation trades fewer growth operations for larger final capacities
on this fixture: the profiled whole-process heap peak rose from 548,951
to 549,655 bytes. It does not demonstrate a smaller heap. The final
fresh-process RSS companion, with 1,500 ms warmup and measurement, recorded
12,336 KiB for Rust and 428,288 KiB for Java (one sample each). RSS includes
runtime/JIT, parsed trace, shared bytes and allocator/GC retention; it
does not isolate retained enumerator state, and JVM settings affect it.

All four result digests matched across engines and repetitions, including
the independent memory companion:

| Operation | Digest |
|---|---|
| Ceiling | `20930ec304b023e5` |
| Exact | `611f110ec1fa3854` |
| Floor | `a2105f4058824220` |
| Next | `d8b393cd0511b20e` |

Fixture SHA-256:
`0363b0f772a532c22216468429d6e3e84bd64cca1309496404941246ce918255`.
The timing twins emit canonical tab-separated checks, and the harness
requires all four checks in every engine and repetition. The gate's report
tests reject missing checks, duplicate rows and adversarial digest mismatches.
This proves report completeness and comparison, not the algorithms' correctness;
`GenIntsRefFstEnum` separately saves and reloads Java FSTs with list,
binary-search, direct-addressing, continuous and BYTE2 encodings, and the
Rust differential test checks all traversal/seek traces. Boundary,
corruption and failed-position recovery units cover both wrappers.

```sh
scripts/docker-test.sh scripts/bench-micro.sh --bench ints_ref_fst_enum \
  --warmup-ms 3000 --measure-ms 3000 --reps 3
scripts/docker-test.sh bash scripts/bench-ints-fst-memory.sh \
  --warmup-ms 1500 --measure-ms 1500
scripts/docker-test.sh bash -c 'env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmemusage.so \
  MICRO_CASE=ints_enum_next MICRO_WARMUP_MS=500 MICRO_MEASURE_MS=1500 \
  taskset -c 0,1 "$CARGO_TARGET_DIR/release/micro" ints_ref_fst_enum'
```
