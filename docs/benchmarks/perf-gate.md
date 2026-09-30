# The nightly performance gate (M6 T6.2)

[M1](../milestones/m1-performance-gate.md)'s harness, made a standing gate:
`.github/workflows/nightly-perf.yml` runs the M1 query mix
(`benchmarks/queries.tsv`) on both corpora (`merged`, `segmented`, 1M
documents from `scripts/bench-corpus.sh`) every night, through
`scripts/bench-gate.py`, which wraps `scripts/bench-compare.sh`.

## What fails the build

| Check | Rule |
|---|---|
| recall | the engines disagree on a query's hit count or top-hit set, or their top scores differ by more than 1e-5 |
| slower than Lucene | a query's Rust/Lucene qps ratio is below 1.0 |
| regression | a query's ratio fell more than 10% below the baseline recorded on the same machine class (`benchmarks/baseline/ci/<corpus>.tsv` for the nightly job, `benchmarks/baseline/local/` for the development machine) |

The gate compares ratios, not qps. Both engines run in the same job on the
same machine, so the runner's own speed cancels out. A query that fails on
timing is measured again, both engines, up to twice more, and fails only if
its best ratio still fails. Runs of the same build move by about ±9% per
query, and a single noisy pass must not fail a night.

Every run appends its ratios to `perf-history-<corpus>.jsonl`. The file is
carried from night to night in the Actions cache, failed runs included, and
uploaded as the run's artifact. A slide shows up there as a trend before it
crosses a threshold.

### Why this bar and not M1's 1.5×

M1's bar is "≥1.5× on at least 80% of queries, none slower". It is not met on
the current build (numbers below). Gating on it would fail every night, and a
gate that is always red is ignored. The gate instead holds what the build does
meet and must keep: nothing slower than Lucene, and nothing slower than it was.
The 1.5×/80% target stays recorded as not met.

## Baseline

`benchmarks/baseline/local/{merged,segmented}.tsv`, recorded with
`bench-gate.py --update-baseline` on the machine in
[`environment.md`](environment.md) (2 pinned cores; M6 build):

| corpus | queries | median ratio | minimum | ≥1.5× | slower than Lucene | recall mismatches |
|---|---|---|---|---|---|---|
| merged | 87 | 1.95× | 1.06× | 65 (75%) | 0 | 0 |
| segmented | 87 | 2.00× | 1.07× | 64 (74%) | 0 | 0 |

**M1's bar (≥1.5× on 80%) is not met**: 75% and 74%. Before this work it was
52% and 55%, with 14 queries slower than Lucene. Getting every query to at
least 1.0× took these changes:
- regexp/terms unions walk bit-set blocks a word at a time, in windows sized
  to what is still needed (`regexp t1[0-9]` 0.39× → 1.7×);
- fuzzy expansion is cheaper: sort keys worked out once, a queue lookup by
  binary search, and no `memcmp` per visited term;
- `TermScorer` scores a batch of documents at once (a term beside phrases);
- phrase positions are read in bulk;
- the blended rewrite's bit set is built from a top-16 selection with blocks
  ORed in;
- a cached filter is tested inline;
- a specialised scorer handles a cached required set plus one optional term
  (`+terms ?term` 1.0× → 2.4×).

Where the remaining quarter goes -- per-block costs in term and boolean
top-k, position decoding in phrases, term expansion in regexps -- is measured
in [`slow-tail-2026-09.md`](slow-tail-2026-09.md).

The first two gate runs on the new build each failed one query, q68 and then
q64, both at 0.92–1.00× after three passes. Those were the last two fixes.

## Negative control

`bench-gate.py --negative-control 2` on `merged` runs every Rust query twice
per timed iteration, a known 2× regression. The gate failed it: 47 queries
were below 1.0× on the first pass and 45 still were after two re-measures,
each still below 1.0× (q01 at 0.95×, q89 at 0.80×). The script therefore exited 0 ("negative
control caught").

The queries that passed anyway were those more than 2× faster than Lucene. A
2× slowdown leaves them above 1.0×, and with no baseline recorded yet only the
"slower than Lucene" rule applied. A second run against the local baseline,
restricted to the 38 queries it records above 2.2×, showed the regression rule
catching them. All 38 failed: 33 as regressed (q31 from 169.7× to 86.8×, q22
from 2.20× to 1.05×) and 5 as slower than Lucene.

Re-run it by hand with the workflow's `negative_control` input, and whenever
the gate script changes.

## What it cannot catch

- A regression smaller than 10% on a query, or a slow drift of a few percent
  a night. The history shows those; the gate does not fail on them.
- A slowdown both engines share. Both run on the same machine and corpus,
  so a slower machine, or a change that slows both, leaves the ratio unchanged.
- A query shape outside `benchmarks/queries.tsv`. The REST-level shapes are
  measured separately (`docs/milestones/m5-6-native-read.md`).
- A regression on a machine class with no baseline. Ratios vary with the CPU,
  so the development machine's baselines (`benchmarks/baseline/local/`) do
  not gate the hosted runner. Until a runner-recorded baseline is committed
  to `benchmarks/baseline/ci/`, the nightly job applies only the
  slower-than-Lucene and recall rules. To record one, run the workflow with
  `update_baseline` and commit the artifact's `benchmarks/baseline/ci/*.tsv`.
  Do the same after a runner change.
