# BlockScoreQueryWrapper

The bounded pair uses Lucene 10.5.0's actual `BlockScoreQueryWrapper` and the
feature-only Rust scorer over the same eight sparse documents and constant-score
boosts as `GenBlockScoreQueryWrapper`. Four block lengths (1, 2, 3, 7) cover
sentinel-aligned blocks and partial final blocks.

Run through `scripts/docker-test.sh bash scripts/bench-block-score-query.sh`.
The `buffer` cases construct a scorer, buffer its input, then consume all matches
and the terminal sentinel. The `bounds` cases construct 256 scorers before each
timer, then perform eight shallow advances and sixteen maxima per scorer.
Construction/destruction are excluded only from `bounds`. Index creation and
query/weight preparation are excluded from both cases. Complete per-batch result
digests are checked in every warmup and measured iteration and across engines.
Both engines hash the current document after each bounds operation, obtain one
iterator per buffer scorer, and publish their final sink after the timer.

This measures the wrapper/scorer seam and result checking. The preparation seams
are different: Rust builds each child scorer tree from its prepared query;
Java's prepared `Weight` retains per-query work. Rust's buffer timer includes
scorer `Drop`; Java makes scorers unreachable and may collect them later. Java
GC occurring within a measured batch is included, but cleanup outside it is not.
There is no whole-search or isolated allocation speedup claim.

## Measurements

Measured on October 25, 2026 through the capped container, pinned to CPUs 0,1,
with a load gate of 1.5, 500 ms warmup and 1000 ms measurement per case, three
alternating Rust/Java repetitions. The table gives median ns/scorer and Java/Rust
ratios. Ranges are minimum to maximum ns across the three repetitions. Earlier
provisional runs used repeated Java iterator creation and different bounds sinks;
those results are excluded.

| Case | Faithful Rust median [range] | Faithful Java median [range] | Ratio | Selected Rust median [range] | Selected Java median [range] | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| bounds 1 | 171.214 [170.821–177.497] | 233.000 [229.993–252.007] | 1.36 | 172.454 [167.912–173.788] | 237.232 [220.107–261.935] | 1.38 |
| bounds 2 | 186.652 [184.825–191.703] | 240.140 [232.525–253.837] | 1.29 | 187.192 [182.612–188.554] | 260.468 [253.931–270.523] | 1.39 |
| bounds 3 | 197.596 [197.084–197.650] | 255.856 [253.372–269.457] | 1.29 | 195.638 [193.288–201.876] | 254.166 [253.173–261.157] | 1.30 |
| bounds 7 | 248.491 [246.859–251.664] | 289.270 [286.030–307.201] | 1.16 | 247.445 [244.512–257.239] | 287.478 [280.603–288.026] | 1.16 |
| buffer 1 | 7061.825 [6954.154–7093.267] | 4683.760 [4585.992–4926.541] | 0.66 | 7065.689 [7006.126–7074.348] | 4844.888 [4452.287–5323.241] | 0.69 |
| buffer 2 | 7157.241 [7154.401–7173.672] | 4599.541 [4379.613–4748.491] | 0.64 | 7173.529 [7052.072–7236.012] | 4775.394 [4438.334–4902.536] | 0.67 |
| buffer 3 | 7223.766 [7149.670–7288.300] | 4466.839 [4435.422–4602.299] | 0.62 | 7107.218 [7080.692–7151.109] | 4559.472 [4444.598–5029.597] | 0.64 |
| buffer 7 | 7194.197 [7107.729–7257.055] | 4528.812 [4510.093–4627.959] | 0.63 | 7135.381 [6969.038–7212.606] | 4458.516 [4447.119–4720.792] | 0.62 |

Stage 1 followed Java's parallel document/score arrays with leading -1 and
trailing NO_MORE_DOCS sentinels. Its vectors retained capacity for 16 values
rather than the fixture's ten. After the faithful run, stage 3 converted the
finished arrays to exact boxed slices and borrowed the existing query at
`Leaf.scorer`, preserving its asserting wrapper, instead of cloning the query
there. The same fixture and regression tests pass. The selected timing changes
are within observed spread; the measured improvement is retained memory.

The buffer cases remain below 1.0. This pair includes the existing scorer-tree
construction seam and deterministic Rust destruction described above, so it does
not isolate time spent growing the wrapper's two arrays. Removing the query
clone did not resolve that difference. The slowdown's cause has not been
profiled. Further work should profile scorer construction and make preparation
and cleanup scopes comparable before choosing an optimization. The
prebuilt wrapper's bounds cases are 1.16–1.39 times Java on this workload.

## Retained memory and validation

For eight matches plus two sentinels, the faithful Rust scorer retained 192 bytes:
64 bytes of scorer fields and 128 bytes of vector capacity. The selected scorer
retains 128 bytes: 48 bytes of fields and 80 bytes of exact array payload. Java's
`RamUsageEstimator` reports 32 bytes of scorer fields and 112 bytes for its two
arrays, totalling 144 bytes with compressed references. These are narrow retained
scorer-and-arrays scopes. They exclude outer handles, iterator/supplier objects,
Weight/query/index state, temporary growth allocations, allocator metadata,
fragmentation, and Java GC bookkeeping. They are not process RSS or peak memory.

The Java generator emits 1,081 rows covering equality, query strings, visitors,
cacheability, no-score modes, scorer iteration, scores, shallow bounds, maxima,
rewriting, empty matches and invalid block lengths. Selected validation passed:
three fixture tests, six scorer tests (including exact 128-byte memory accounting),
the supplier threshold regression, five score-mode-filter tests, 17 SimpleText
tests, nine shared test-support tests and 56 asserting tests. All eight paired
benchmark digests matched across engines in every repetition. The parent wave
runs the full gate and independent semantic review.
