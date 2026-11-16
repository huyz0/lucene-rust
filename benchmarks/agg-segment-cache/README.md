# Aggregation segment cache benchmark

Results and method: [docs/benchmarks/agg-segment-cache.md](../../docs/benchmarks/agg-segment-cache.md).

1. Loghub lines, the first 250 MB of each set (openssh is smaller):

       mkdir -p data && for s in bgl hdfs openssh thunderbird; do
         curl -sL https://huggingface.co/datasets/bolu61/loghub_2/resolve/main/data/$s.txt | head -c 250000000 > data/$s.txt
       done

2. Events: `python3 gen.py data WORK/corpus` (6,454,808 events: 6,254,808 loaded, 200,000 for
   the live rounds).
3. Libraries: `liblucene_ffi.so` from main and from this branch
   (`cargo build --profile dist -p lucene-ffi --features mimalloc`), and the plugin zip built
   once (`gradle -p opensearch-plugin bundlePlugin`). The nodes are plain processes
   (`opensearch-plugin/e2e/nodectl.py`), no Docker.
4. `./run.sh WORK main.so cache.so` (port 9250 free, about 15 minutes for all three configurations).
