# HNSW refresh with vectors buffered

A refresh flushes what the writer buffered. Lucene's
`Lucene99HnswVectorsWriter` inserts each vector into its field's HNSW graph
as the document is added (`FieldWriter.addValue`), so the flush only writes
a graph that is already built. The port used to keep the vectors and build
the whole graph at the flush, so every refresh with vectors buffered paid
for the full graph build.

The concurrent writer now builds the graph as documents arrive, on the
indexing thread, for every buffer: realtime search already kept such a graph
(`BufferGraph`), and a buffer without it now keeps one only for its flush.
The flush writes that graph. Under an index sort it is renumbered to the
sorted ordinals, as `writeSortingField` does in Lucene, instead of being
rebuilt over the sorted vectors. The graph's heap counts against the RAM
buffer, as Lucene's `ramBytesUsed` counts it.

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
| plain, this change | 29.4 / 30.9 | 26 / 27 | 156 / 176 | 356 / 324 | 14 (100) / 15 (112) |
| plain, Lucene 10.5.2 | 54.7 / 55.8 | 46 / 47 | 196 / 214 | 687 / 642 | 24 (240) / 24 (248) |
| tenant sort, main | 26.9 | 5 | 4,950 | 7,029 | 2 (31) |
| tenant sort, this change | 29.8 / 29.6 | 25 / 25 | 219 / 218 | 389 / 432 | 13 (91) / 13 (89) |
| tenant sort, Lucene 10.5.2 | 60.3 / 53.8 | 47 / 44 | 279 / 241 | 783 / 726 | 25 (264) / 23 (226) |

- Refresh drops from about 5 s to about 0.2 s, at or below stock's.
- Indexing takes about 30% longer than on main, because the graph work moves
  from the flush to the indexing threads. It is still about 1.8x faster than
  stock.
- More refreshes now land in the same time, so more segments are flushed and
  natural merging does more work. Stock does the same.
- Tenant-filtered top-10 recall stays at 0.9999-1.0. Force merge is
  unchanged, since merges build their graphs as before.

Inserting from a background task with up to 512 vectors waiting gave
765 ms refreshes and no faster indexing. Inserting from the indexing
thread is what Lucene does, and it measured best.
