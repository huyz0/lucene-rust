# Native SimpleText phrase scoring

`SimpleTextPhraseMicro.java` and `micro_stphrase.rs` compare native
SimpleText phrase execution with Lucene 10.5.0 on the committed
`fixtures/data/simpletext_phrase_scorer/deep/merged/index`. Reader opening
and norms setup are outside timing. Both engines use ClassicSimilarity,
disable the query cache, and search the same segment with its deletions.

The Rust harness supplies the native segment reader to the existing scorer
tree through `OpenSegment`; it does not translate text postings into binary
postings. The general `DirectoryReader.open_segments` search guard remains.
This is a single-segment comparison: the scorer uses that segment's native
statistics, and the harness does not establish reader-wide text query routing.

There are six cases: exact `"alpha beta"`, sloppy `"beta alpha"~2`, and
repeated `"alpha alpha"~3`, each collecting either every score or the top ten
with a total-hits threshold of zero. Each case checks an untimed digest of
document IDs and raw score bits before timing. Exhaustive digests also
include the hit count; top-ten digests include the number of returned hits.
The benchmark report rejects different digests between engines.

The broader differential test is
`native_phrases_match_java_documents_scores_bounds_and_competitive_iteration`,
using `GenSimpleTextPhraseScorer`. It additionally covers BM25, boosts,
unscored execution, missing fields and terms, explicit positions, repeat
groups at equal positions in both lexical orders, equal-frequency terms with
different skip boundaries at target 154, score bounds at all four impacts levels,
competitive iteration, and Java's errors on fields without positions.

Three alternating repetitions used 3,000 ms warmup and 3,000 ms measurement
per case, pinned to CPUs 0 and 1 in the capped container. Each case ran in
a fresh process. The shared index has 768 documents, three deleted.

| Case | Rust ns/search | Java ns/search | Java/Rust | Noise |
|---|---:|---:|---:|---:|
| Exact, all scores | 356,375.011 | 1,100,225.843 | 3.09x | 1.05x |
| Exact, top ten | 97,301.778 | 408,191.023 | 4.20x | 1.05x |
| Sloppy, all scores | 366,109.452 | 1,130,183.088 | 3.09x | 1.04x |
| Sloppy, top ten | 365,826.958 | 1,117,065.088 | 3.05x | 1.04x |
| Repeated term, all scores | 384,126.185 | 1,063,600.642 | 2.77x | 1.18x |
| Repeated term, top ten | 455,108.622 | 1,155,013.956 | 2.54x | 1.04x |

Every result digest matched across engines and repetitions. The median
ratio was 3.07x; the run's noise floor was 1.04x. All six ratios are outside
that floor. The faithful native adapter meets the timing comparison;
no algorithm optimisation was needed. Peak process RSS in the memory
companion was 12,872 KiB for Rust and 275,184 KiB for Java. All six digests
also matched in that fresh runner invocation.

Reproduce the timing and peak-process-memory comparisons with:

```sh
scripts/docker-test.sh scripts/bench-micro.sh --bench stphrase \
  --warmup-ms 3000 --measure-ms 3000 --reps 3
scripts/docker-test.sh bash scripts/bench-stphrase-memory.sh
```

The memory companion reports peak process RSS for each engine in a fresh
process. It includes the runtime, reader, cached terms, norms and transient
scorer state. It does not isolate scorer allocations, demonstrate scaling
with index size, or cover the memory cost of multiple segments. Timing
results do not cover BM25, other query families, or general codec routing.
