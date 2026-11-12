# Aggregation segment cache benchmark

Results and method: [docs/benchmarks/agg-segment-cache.md](../../docs/benchmarks/agg-segment-cache.md).

1. Loghub lines, the first 250 MB of each set (openssh is smaller):

       mkdir -p data && for s in bgl hdfs openssh thunderbird; do
         curl -sL https://huggingface.co/datasets/bolu61/loghub_2/resolve/main/data/$s.txt | head -c 250000000 > data/$s.txt
       done

2. Events: `python3 gen.py data WORK/corpus` (6,454,808 events: 6,254,808 loaded, 200,000 for
   the live rounds).
3. Images: `lucene-rust-opensearch:main` from main and `lucene-rust-opensearch:cache` from this
   branch, built as `scripts/bench-opensearch.sh` builds its image, and
   `opensearch-stock-min:3.9.0`.
4. `./run.sh WORK lucene-rust-opensearch:main lucene-rust-opensearch:cache` (needs Docker, port
   9250 free, about 15 minutes for all three configurations).
