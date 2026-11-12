# Filtered sort on merged segments

A selective filter sorted by a time field (`src=openssh`, about 10% of the
6.45M Loghub lines, all of them in the oldest 40% of the time range) on the
5-segment layout of [newest-n.md](newest-n.md) ran at about 2.5x Lucene's time.
The bounded newest-N search gives up on it at once (the newest matches lie
behind most of the index) and hands it to the ordinary sorted search.

## Where the time went

The 639k matches all sit in the one merged segment, whose documents are
time-ordered runs. Read against a descending sort, almost every match beats the
queue's bottom: Lucene and this port both visit all 639k and both make the same
263k queue replacements (counted on both sides). The gap was the cost per hit:

- the queue's sift compared through the general per-key comparator, two loads
  deep per level;
- a varying-bits-per-value doc-values read went through three out-of-line calls
  (and built a `DirectReader`, with its width check, per value);
- the scorer was asked for its run of consecutive matches on every document,
  each a bit-set scan, though the runs here are one or two documents long;
- the doc-values skip index's competitive iterator was walked document by
  document as the lead (`contains` on every one of the segment's 5.5M
  documents), because it cannot say how many documents it holds.

## The change

- `HitQueue`: with one plain numeric key, each entry carries its key (the
  value, complemented for a reversed sort), and a level of the sift is one load
  and a branch-free 128-bit comparison of `(key, doc)`. The new top's key and
  document go into the sift in registers. Any other sort is unchanged.
- `NumericReader::block_value`: a dense varying-bits-per-value field read
  inline from the block the last read decoded, with `direct_reader::get_fast`.
- The run question is kept only while runs of 16 or more documents keep
  coming; shorter runs count as misses.
- The competitive iterator leads (membership tests) only when its cost is below
  the scorer's.
- Newest-N's fallback (the whole pass after the slices) no longer runs the
  collector: its total is counted separately, so it reads every match after the
  last bound into a buffer, cut back to the best `n` by selection whenever it
  fills, the worst kept becoming the bar. Same hits, same order, ties broken by
  document.

## Results (warm, ms, same machine and harness as newest-n.md)

`benchmarks/newest-n/results/filtered-sort/` holds all 63 shapes for Lucene
10.5.1, main (`3fe63460`) and this change, in both layouts. Four new shapes
stress the fallback: `src=openssh ts asc`, and ranges on another time field
(`tsec`, the old half, the middle third) sorted by `ts`. **All 126 shape runs
return the same hits (doc and sort value, in order) as Lucene.**

5 segments:

| shape | N | Lucene | main | this change |
|---|---:|---:|---:|---:|
| src=openssh ts desc | 10 | 16.8 | 38.2 | 9.4 |
| src=openssh ts desc | 100 | 34.3 | 54.6 | 9.6 |
| src=openssh ts desc | 1000 | 48.0 | 66.5 | 9.6 |
| src=openssh ts_s(skipper) desc | 10 | 12.3 | 50.9 | 9.2 |
| src=openssh ts_s(skipper) desc | 1000 | 43.6 | 77.3 | 9.6 |
| src=hdfs ts desc | 1000 | 138 | 24.7 | 13.9 |
| match_all ts asc | 10 | 65.7 | 14.1 | 5.7 |
| match_all ts asc | 1000 | 66.9 | 79.0 | 31.8 |
| tsec(old half) ts desc | 100 | 178 | 121 | 66 |
| tsec(middle third) ts desc | 100 | 123 | 141 | 83 |
| tsec(middle third) ts asc | 100 | 28.0 | 30.8 | 12.9 |
| src=openssh ts asc | 10 | 4.2 | 8.8 | 5.1 |
| match_all tmiss desc missing_first | 1000 | 85.6 | 96.7 | 94.2 |

26 segments: every filtered shape is 1.1x to 2.1x faster than main (for
example `src=openssh ts_s(skipper) desc` N=10 5.4 -> 2.8, `src=hdfs ts desc`
N=1000 24.9 -> 13.8); the sub-millisecond shapes are within 0.1 ms of main.

The two shapes still behind Lucene are `src=openssh ts asc` on 5 segments
(5.1 vs 4.2 ms, from 8.8 on main) and `tmiss desc missing_first` N=1000 (94 vs
86, as on main): both are ordinary-path searches whose cost is visiting the
matches, not the queue.
