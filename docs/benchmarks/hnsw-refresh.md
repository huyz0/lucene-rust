# HNSW refresh with vectors buffered

A refresh flushes what the writer buffered. Lucene's
`Lucene99HnswVectorsWriter` inserts each vector into its field's HNSW graph
as the document is added (`FieldWriter.addValue`), so the flush only writes
a graph that is already built. The port used to keep the vectors and build
the whole graph at the flush, so every refresh with vectors buffered paid
for the full graph build.

The concurrent writer now builds the graph as documents arrive, on the
indexing thread, for every buffer. A buffer with realtime search keeps the
concurrent `BufferGraph` it already had, which searches read while it
grows. A buffer without it keeps a `FlushGraph`: `HnswGraphBuilder` itself,
over a plain list of the vectors the documents hold (shared, not copied),
with no locks. The flush writes the kept graph. Under an index sort it is
renumbered to the sorted ordinals, as `writeSortingField` does in Lucene,
instead of being rebuilt over the sorted vectors. The graph's heap counts
against the RAM buffer, as Lucene's `ramBytesUsed` counts it.

The files stay the same: the kept graph is the one `HnswGraphBuilder`
builds over the same vectors with the same seed. Tests check both cases
byte for byte, unsorted and sorted.

## Results

500k x 96-d cosine vectors with 1000 Zipf tenants, written by 4 threads with
a reader refreshed every second. 32 MB buffer, `TieredMergePolicy` and
`ConcurrentMergeScheduler` defaults, then a commit. Run with
`partition_cost index` and `PartitionCostBench index`
(`scripts/bench-partition-cost.sh vectors`). The machine is a 4-vCPU VM.
Each cell is two runs, and main is one run.

| layout, engine | index s | refreshes | refresh mean ms | refresh p99 ms | merges (s) |
|---|---:|---:|---:|---:|---:|
| plain, main | 22.5 | 4 | 5,592 | 7,625 | 2 (40) |
| plain, this change | 25.7 / 27.1 | 23 / 24 | 131 / 172 | 256 / 308 | 12 (100) / 13 (97) |
| plain, Lucene 10.5.2 | 54.7 / 55.8 | 46 / 47 | 196 / 214 | 687 / 642 | 24 (240) / 24 (248) |
| tenant sort, main | 26.9 | 5 | 4,950 | 7,029 | 2 (31) |
| tenant sort, this change | 27.7 / 27.3 | 23 / 23 | 203 / 204 | 380 / 362 | 13 (97) / 12 (86) |
| tenant sort, Lucene 10.5.2 | 60.3 / 53.8 | 47 / 44 | 279 / 241 | 783 / 726 | 25 (264) / 23 (226) |

- Refresh drops from about 5 s to about 0.2 s, at or below stock's.
- Indexing takes 15-20% longer than on main, because the graph work moves
  from the flush to the indexing threads. It is still about 2x faster than
  stock.
- More refreshes now land in the same time, so more segments are flushed and
  natural merging does more work. Stock does the same.
- Tenant-filtered top-10 recall stays at 0.9999-1.0. Force merge is
  unchanged, since merges build their graphs as before.

## What else was measured

- Inserting from a background task, with up to 512 vectors waiting, gave
  765 ms refreshes and no faster indexing.
- Inserting through realtime's `BufferGraph` from the indexing thread gave
  the same refreshes but 29-31 s indexing. In a 60k-vector build it takes
  10.8 s against `HnswGraphBuilder`'s 8.2 s. The cost is its per-node locks
  and its lock-free vector table, which a buffer nobody searches does not
  need; `FlushGraph` takes 8.8 s.
- Copying the vectors into contiguous storage would save about 7% more of
  the insert time, but it holds every buffered vector twice, so it was left
  out.
