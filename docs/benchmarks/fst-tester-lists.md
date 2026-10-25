# FST tester list outputs

The pair executes Lucene 10.5.0's actual `FSTTester` with `ListOfOutputs`,
including list expansion, compilation, save/read, complete pair verification,
random accepted words, seeks and mixed enumeration. Each operation handles one
of six seeded cases per input mode; batches include all six. Both twins copy
every input/pair/list container, consume all saved bytes and enumerated values,
compiler counts and RNG tails, and require matching digests throughout warmup
and measurement. Java assertions and `tests.multiplier=1` remain enabled.

Three alternating repetitions use CPUs 0–1, a 128 MiB Java heap, 1,000 ms warmup
and 1,500 ms measurement. Fixtures and Java's reused ByteBuffersDirectory are
prepared outside timing. Rust models off-heap and IOContext paths as owned
save/read byte round trips, preserving the Java random draws.

| Case | Rust ns/tester | Java ns/tester | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| UTF-8 | 394,097.024 | 547,371.945 | 1.39x | 1.07x |
| UTF-32 | 406,976.329 | 423,311.597 | 1.04x | 1.07x |

The UTF-32 result is indistinguishable within measured noise. The faithful
baseline measured 431,408.758/523,290.907 ns for UTF-8 and
460,215.904/389,013.348 ns for UTF-32 (Rust/Java), leaving UTF-32 at 0.85x.
Temporary aggregate phase timers attributed 57.0% of UTF-32 tester time to
random walks, 33.1% to mixed enumeration and 7.8% to seeks. Separately timed
UTF-32 conversion accounted for about 4.1% including timer overhead. These are
instrumented elapsed phase times, not a sampling CPU profile or allocation trace.
Instrumentation was removed before the reported paired measurements.

The retained optimization reuses one outgoing-arc vector across random walks
instead of allocating it for every accepted word. It decodes every arc in the
same order and preserves the random choices, assertions, outputs and error
propagation. No decoded arcs or outputs are retained between walks. Rust medians
improved 1.09x for UTF-8 and 1.13x for UTF-32 against the baseline, while Java
timings also varied between runs. The paired table is the current comparison;
it does not attribute all cross-run ratio changes to this optimization.

The tester and its reachable input-container/scalar graph occupy 968 bytes Rust
and 1,856 bytes Java. Java boxed scalar identities are counted once, including
shared cached Long objects; backing array capacities match the timed copy
constructors. RNG, directory, output algebra, compiled FST, compiler/verification
scratch, allocator metadata and runtime are excluded. Reused arc capacity is
verification scratch, not part of that graph. No total or peak-memory claim is
made. This is the complete tester workload on the small mixed-list fixture,
not a general FST compilation benchmark.

`GenFstTesterLists` independently exercises 108 actual Java tester cases across
six seeds, both input modes and nine scenarios, including scalar/mixed lists,
zero outputs, empty/singleton inputs, shared prefixes, supplementary characters
and invalid inputs. Differential tests assert complete saved bytes, enumeration,
compiler counts, failure status and RNG continuation. The typed Rust
`do_test_lists` entry point replaces Java runtime list detection.

```sh
scripts/docker-test.sh bash scripts/bench-fst-tester-lists.sh 1000 1500
```
