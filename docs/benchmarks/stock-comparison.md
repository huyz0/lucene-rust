# Stock Lucene, stock OpenSearch, the plugin and the core engine

This report puts four things side by side: Apache Lucene 10.6 on its own, stock OpenSearch 3.9.0, the
plugin running inside OpenSearch 3.9.0, and the core engine on its own with no JVM and no OpenSearch.
The gap between the plugin and the core engine is the cost of OpenSearch's own request path, which the
plugin shares with stock.

## Setup

One 4-core, 15 GB cloud machine, one run at a time. The OpenSearch node (bundled JDK 25, 2 GB heap, one
shard, no replica) runs on cores 0-2 and the client on core 3; the in-process harnesses run on cores 0-1. Each side is
measured on its own start of the node, after warm-up, against a single force-merged segment. The client
is one keep-alive HTTP socket, so it adds about 0.1 ms to every REST number. All of it is in the repository:

| Column | What it is | Harness |
|---|---|---|
| Lucene | Apache Lucene 10.6.0 in a plain JVM (`jdk.incubator.vector` on), top 50, no OpenSearch | `benchmarks/java-runner` |
| Core engine | the engine as a library, top 50, no JVM, no OpenSearch, same index files | `benchmarks/rust-runner`, joined with `scripts/bench-compare.sh` |
| OpenSearch | stock OpenSearch 3.9.0, REST round trip, p50 | `benchmarks/native-node/split.py` |
| Plugin | the plugin on the same node with its defaults (direct search on, which answers eligible single-shard searches without OpenSearch's search action), REST round trip, p50; the mean of two runs | `split.py` |
| Plugin, direct off | the plugin with `search.accelerated.direct.enabled: false`, so every request goes through OpenSearch's search action | `split.py` |
| Core in plugin | the time the plugin spent in the engine for the request (query phase plus fetch), read from `/_plugins/accelerated/stats` on the direct-off run | `split.py` |

The Lucene and core-engine columns use a 1M-document synthetic corpus built by
`benchmarks/corpus` (Zipf terms, one segment, 87 query shapes in `benchmarks/queries.tsv`; the engines are
checked to return the same hits and top documents before any ratio counts). The OpenSearch columns use the
corpus from `opensearch-plugin/e2e/bench_nodes.py gen` (1M documents) plus the feature-family indices
of `benchmarks/native-node/families.py`, 195 request shapes. The two corpora share a design but not
the documents, so compare ratios across the two groups, not milliseconds.

Core in plugin is a measurement of the engine's own time inside the node: it includes the foreign-function
crossing (under 1% of a search in the bridge measurement) and excludes everything OpenSearch does around the call.
It is not available for shapes the plugin hands back to OpenSearch.

## Which shapes have which baseline

| Shape family | Lucene | OpenSearch | Plugin | Core engine on its own |
|---|---|---|---|---|
| term, match, bool, phrase, prefix, wildcard, regexp, fuzzy | yes | yes | yes | yes |
| sort, `search_after` | yes | yes | yes | yes |
| indexing, merge | yes (writer) | yes | yes | yes |
| HNSW k-NN | yes (`KnnFloatVectorQuery`, reports in the HNSW report) | yes (lucene engine) | yes | yes |
| aggregations (metrics, buckets, pipeline), joins, scripts, highlight, fetch | no: Lucene has facets and grouping, not OpenSearch's aggregation DSL | yes | yes | only as time inside the plugin |
| neural, hybrid, rerank, semantic fields | no | yes (stock k-NN, neural-search, ml-commons) | yes | only as time inside the plugin |

## Lucene against the core engine (87 shapes, 1M documents, in process)

Geometric mean of queries per second, core engine over Lucene: 2.69x; median 1.81x; 75 of 87 shapes are
faster, 12 slower.

| Shape group | Shapes | Geomean | Slowest |
|---|---:|---:|---:|
| term | 7 | 2.96x | 1.81x |
| `AND` | 5 | 1.93x | 1.76x |
| `OR` and block-max `OR` | 6 | 1.74x | 1.47x |
| phrase | 2 | 1.63x | 1.57x |
| prefix, wildcard | 4 | 5.9x | 4.26x |
| regexp | 10 | 12.5x | 1.34x |
| fuzzy | 1 | 2.14x | |
| sorted (numeric, keyword, score) | 18 | 5.62x | 1.14x |
| composed bool queries | 32 | 1.23x | 0.03x |

The composed boolean queries are where the core engine is slower than Lucene: queries with a `MUST_NOT` clause on a frequent term
(`+t2s -t0` takes 6.8 ms against Lucene's 0.38 ms, `+t0 -wc(t1?)` 7.2 ms against 3.1 ms) and a few
large-union phrase mixes. These are listed as weak spots in [Performance](https://huyz0.github.io/lucene-rust/performance.html#weak-spots). The numeric `dv_sort` shape (`0.26x`)
is the other one. Full per-query results are in the raw files kept with the benchmark records.

## Indexing: Lucene against the core engine (1M log-shaped documents, 4 writer threads, 1 s refresh)

Two alternating runs each.

| | docs/s | CPU seconds | Index MB | Force merge to one segment |
|---|---:|---:|---:|---|
| Lucene 10.6 `IndexWriter` | 78,000-93,000 | 44-51 | 226-228 | 4.8 s, 12.0 CPU s |
| Core engine writer | 202,000-230,000 | 20-21 | 227 | 2.3 s, 2.7 CPU s |

## Indexing: OpenSearch against the plugin (REST, bulk 5,000 x 4 clients)

Third pass on one node (the warm figure), new index each pass; one run per cell, run-to-run spread 5-10%.

| Documents | OpenSearch docs/s | Plugin docs/s | Ratio | OpenSearch CPU s per 1M | Plugin CPU s per 1M |
|---|---:|---:|---:|---:|---:|
| logs, 10 fields, 1M | 68,900 | 97,300 | 1.41x | 41.7 | 26.8 |
| text, 120 words, 1M | 30,800 | 47,300 | 1.53x | 92.7 | 57.9 |
| nested, 5 children, 500k (second pass) | 27,000 | 34,800 | 1.29x | 105.3 | 80.7 |

The first pass on a freshly started node is slower on both sides while the JVM compiles (logs: 40,900 docs/s for OpenSearch,
81,700 for the plugin, which compiles less Java). The earlier run of this harness on a different day measured the stock node faster
on this class of documents (about 95,000 docs/s warm on logs), which gave 1.3x for logs; treat 1.3-1.4x as the
range for warm logs, not as a precise figure.

## k-NN, neural, hybrid and rerank (OpenSearch against the plugin)

Stock's own k-NN, neural-search and ml-commons against the plugin's, with small ONNX models served locally, 4,000 documents,
300 queries per shape, best median of 2 rounds (`benchmarks/native-node/neural-ab.py`, no Docker). Model inference is
the same ONNX runtime on both sides, so shapes dominated by the model show a ratio near 1.

| Shape | OpenSearch (ms) | Plugin (ms) | Ratio |
|---|---:|---:|---:|
| `knn` k=10, precomputed vector | 2.86 | 0.90 | 3.20x |
| `knn` k=10 with a term filter | 2.49 | 0.87 | 2.88x |
| `knn` k=100, size 100 | 19.52 | 9.89 | 1.97x |
| `neural` (embed the query, then k-NN) | 6.24 | 5.44 | 1.15x |
| `neural_sparse` by model | 13.62 | 11.75 | 1.16x |
| `neural_sparse` by analyzer | 3.52 | 1.70 | 2.07x |
| semantic field, `neural` | 3.50 | 2.85 | 1.23x |
| MMR, 50 candidates | 10.32 | 6.08 | 1.70x |
| hybrid `match` + `knn` (min_max) | 6.57 | 3.72 | 1.77x |
| hybrid `match` + `neural` (min_max) | 6.37 | 4.88 | 1.30x |
| hybrid `match` + `knn` (rrf) | 6.34 | 3.65 | 1.74x |
| rerank, top 10, cross-encoder | 20.75 | 11.82 | 1.75x |
| ingest with embedding and sparse encoding (docs/s) | 204.6 | 266.6 | 1.30x |
| ingest into a semantic field (docs/s) | 581.7 | 743.8 | 1.28x |

Geometric mean of the search shapes: 1.75x. The `lucene`-engine HNSW k-NN on 200,000 128-dimension vectors, REST, best of two
rounds: k=10 4.23 against 1.66 ms, k=100 3.99 against 1.90 ms, k=10 with a `_source` fetch 3.19 against 1.72 ms,
k=10 with a keyword filter 6.03 against 1.78 ms, k=10 with a 1% range filter 2.86 against 1.17 ms.
Lucene against the core engine for HNSW on GloVe-100 is in the [HNSW report](hnsw.md) (reused, measured on Lucene 10.5).

## OpenSearch against the plugin, all 195 shapes

Geometric mean of the stock-over-plugin ratio. Shapes the plugin hands back to OpenSearch (20 of 195, mostly
the less common aggregations) run at about 1.0x by construction and are counted in the totals. The two plugin runs with direct search on agree to within 1.4%
on the whole-sweep geometric mean.

| Group | Shapes | Plugin | Plugin, direct off | CPU per request, OpenSearch over plugin |
|---|---:|---:|---:|---:|
| all | 195 | 1.76x | 1.27x | 1.83x |
| run natively | 175 | 1.86x | 1.30x | 1.97x |
| native, stock under 4.5 ms | 70 | 1.77x | 0.98x | 1.92x |
| native, stock 4.5 ms or more | 105 | 1.93x | 1.58x | 2.00x |
| handed back to OpenSearch | 20 | 1.04x | 0.97x | 0.97x |

With defaults, 114 shapes are 1.5x or faster and 28 are slower than stock. The cheap shapes show what direct search is for: with it off, shapes under 4.5 ms of stock time are level with stock (0.98x), because the OpenSearch request path is most of their time; with it on they are 1.77x. The engine's own time (core in plugin) against stock's round trip is 2.79x
over the 173 shapes where it is reported. The median time left in a native-run request outside the engine is 0.4 ms with direct search and 1.5 ms without it.

| Family | Shapes | Plugin vs OpenSearch |
|---|---:|---:|
| floor | 2 | 4.03x |
| totals | 4 | 2.84x |
| exists | 1 | 2.57x |
| sort | 5 | 2.49x |
| fetch | 6 | 2.23x |
| range | 4 | 2.15x |
| prefix | 2 | 2.11x |
| core mix | 63 | 2.06x |
| wildcard | 5 | 1.99x |
| ids | 1 | 1.98x |
| query_string | 7 | 1.95x |
| regexp | 3 | 1.94x |
| rescore | 1 | 1.89x |
| geo | 3 | 1.83x |
| fuzzy | 3 | 1.81x |
| bool | 12 | 1.77x |
| terms | 2 | 1.52x |
| agg | 35 | 1.52x |
| function_score | 4 | 1.36x |
| script | 7 | 1.34x |
| nested | 6 | 1.29x |
| other-types | 4 | 1.15x |
| suggest | 5 | 1.02x |
| highlight | 4 | 1.01x |
| join | 6 | 0.92x |

### Every shape (milliseconds, p50)

Core in plugin is blank where the plugin hands the shape to OpenSearch.

| Shape | OpenSearch | Plugin | Plugin, direct off | Core in plugin | Plugin vs OpenSearch |
|---|---:|---:|---:|---:|---:|
| agg/adjacency_matrix | 33.69 | 12.32 | 12.03 | 11.07 | 2.73x |
| agg/agg on query t0 (dense) | 39.72 | 9.37 | 12.30 | 10.14 | 4.24x |
| agg/auto_date_histogram 50 | 1.45 | 1.59 | 1.70 |  | 0.91x |
| agg/date_histogram day | 7.64 | 3.45 | 3.50 | 1.74 | 2.21x |
| agg/date_histogram hour fixed + tz | 53.95 | 49.24 | 54.19 |  | 1.10x |
| agg/date_range 4 | 1.26 | 1.36 | 1.32 | 0.25 | 0.93x |
| agg/diversified_sampler | 36.46 | 40.32 | 38.20 |  | 0.90x |
| agg/extended_stats price | 23.73 | 5.15 | 6.76 | 6.03 | 4.61x |
| agg/filter + filters | 24.32 | 10.60 | 10.87 | 9.99 | 2.29x |
| agg/geo_bounds + geo_centroid | 71.56 | 54.65 | 61.60 |  | 1.31x |
| agg/geo_distance agg 3 rings | 113.01 | 103.14 | 109.33 |  | 1.10x |
| agg/geohash_grid precision 3 | 95.72 | 105.09 | 93.55 |  | 0.91x |
| agg/geotile_grid zoom 5 | 85.05 | 78.82 | 72.77 |  | 1.08x |
| agg/histogram num interval 10000 | 27.24 | 10.54 | 13.25 | 11.25 | 2.59x |
| agg/matrix_stats | 46.66 | 40.81 | 42.59 |  | 1.14x |
| agg/median_absolute_deviation | 212.25 | 175.33 | 203.59 | 202.04 | 1.21x |
| agg/missing + global | 20.65 | 9.27 | 10.34 | 8.72 | 2.23x |
| agg/multi_terms category+tag | 98.75 | 102.10 | 145.54 |  | 0.97x |
| agg/percentile_ranks price | 105.47 | 103.51 | 105.91 | 105.42 | 1.02x |
| agg/pipeline date_hist > sum > cumulative_sum | 20.49 | 17.09 | 14.77 | 13.41 | 1.20x |
| agg/range price 5 ranges | 1.23 | 1.33 | 1.50 | 0.24 | 0.93x |
| agg/rare_terms tag | 1.78 | 1.90 | 2.19 |  | 0.94x |
| agg/sampler > terms | 23.29 | 19.14 | 21.83 |  | 1.22x |
| agg/significant_terms tag (fg t8) | 15.25 | 21.86 | 21.72 | 20.72 | 0.70x |
| agg/significant_text body (fg t8) | 14687.19 | 18387.45 | 21026.83 | 20914.43 | 0.80x |
| agg/terms > avg + sibling avg_bucket | 37.64 | 11.77 | 12.02 | 11.30 | 3.20x |
| agg/terms > bucket_sort | 46.42 | 14.35 | 12.86 | 11.00 | 3.23x |
| agg/terms include/exclude regex | 14.40 | 7.55 | 7.06 | 6.01 | 1.91x |
| agg/terms min_doc_count + shard_size | 1.38 | 0.57 | 1.37 | 0.11 | 2.42x |
| agg/terms on ip-less num > cardinality | 57.01 | 17.95 | 18.87 | 18.04 | 3.18x |
| agg/terms order by sub-agg avg | 39.24 | 40.30 | 36.04 |  | 0.97x |
| agg/terms size 1000 on tag | 4.07 | 2.10 | 3.44 | 0.85 | 1.94x |
| agg/value_count + min + max + avg | 25.26 | 5.79 | 9.44 | 7.91 | 4.36x |
| agg/variable_width_histogram | 104.48 | 89.30 | 100.95 |  | 1.17x |
| agg/weighted_avg | 30.22 | 31.18 | 28.77 |  | 0.97x |
| bool/bool 4 should minimum_should_match 2 | 42.77 | 13.57 | 18.38 | 17.56 | 3.15x |
| bool/bool must+should+must_not+filter | 6.51 | 5.78 | 6.33 | 5.15 | 1.13x |
| bool/boosting | 38.57 | 38.40 | 42.28 |  | 1.00x |
| bool/combined_fields | 38.14 | 19.71 | 31.19 | 26.96 | 1.93x |
| bool/common-term OR 8 terms | 27.14 | 14.38 | 22.55 | 19.90 | 1.89x |
| bool/constant_score filter | 1.22 | 0.65 | 1.45 | 0.20 | 1.88x |
| bool/dis_max 2 fields | 40.79 | 8.23 | 9.64 | 8.35 | 4.96x |
| bool/match bool_prefix | 12.94 | 8.97 | 15.75 | 12.88 | 1.44x |
| bool/more_like_this | 5.28 | 2.95 | 4.59 | 3.33 | 1.79x |
| bool/multi_match cross_fields | 17.91 | 11.79 | 15.14 | 14.18 | 1.52x |
| bool/multi_match most_fields | 4.15 | 2.41 | 4.01 | 2.34 | 1.72x |
| bool/multi_match phrase_prefix | 266.55 | 227.37 | 240.25 | 247.27 | 1.17x |
| core/bool match + range filter | 3.09 | 2.24 | 3.12 | 1.36 | 1.38x |
| core/cardinality num | 17.11 | 10.90 | 11.97 | 10.84 | 1.57x |
| core/cardinality tag | 9.32 | 1.68 | 1.69 | 0.36 | 5.56x |
| core/composite category 10 | 7.39 | 3.35 | 3.67 | 2.15 | 2.20x |
| core/composite category+tag | 3.54 | 4.50 | 4.32 | 1.98 | 0.79x |
| core/composite category+tag 100 | 3.19 | 2.54 | 2.94 | 1.16 | 1.25x |
| core/composite category+tag 100 > avg price | 4.92 | 3.87 | 4.14 | 2.38 | 1.27x |
| core/composite category+tag 100 after | 3.37 | 2.41 | 2.49 | 0.91 | 1.40x |
| core/composite category+tag > avg price | 5.34 | 5.21 | 6.26 | 3.56 | 1.02x |
| core/composite category+tag, filtered | 26.39 | 11.87 | 10.99 | 8.69 | 2.22x |
| core/composite category+tag, size 10 | 3.12 | 3.02 | 2.37 | 0.90 | 1.03x |
| core/composite num+category (long first) | 1.62 | 2.28 | 2.12 | 0.47 | 0.71x |
| core/composite tag+category (first field > size) | 12.50 | 9.64 | 8.15 | 6.19 | 1.30x |
| core/composite tag+category after | 3.49 | 3.17 | 3.30 | 1.64 | 1.10x |
| core/composite ts day+category (date first) | 2.42 | 2.97 | 3.50 | 1.57 | 0.81x |
| core/count, track_total_hits | 1.27 | 0.28 | 1.93 | 0.11 | 4.55x |
| core/date_histogram by month | 1.72 | 1.68 | 1.70 | 0.43 | 1.02x |
| core/date_histogram day > terms category | 102.84 | 31.66 | 36.99 | 31.34 | 3.25x |
| core/filters 3 > stats price | 58.45 | 31.74 | 31.50 | 30.46 | 1.84x |
| core/geo_bounding_box filter | 1.78 | 0.52 | 1.77 | 0.27 | 3.45x |
| core/geo_distance filter | 2.37 | 0.99 | 1.94 | 0.51 | 2.40x |
| core/histogram price 10 > stats num | 62.28 | 20.74 | 20.30 | 18.51 | 3.00x |
| core/intervals, any_of | 236.56 | 178.99 | 194.10 | 196.87 | 1.32x |
| core/intervals, ordered match with gaps | 317.81 | 265.20 | 309.44 | 306.07 | 1.20x |
| core/match AND, 2 frequent | 8.15 | 4.51 | 4.78 | 3.02 | 1.81x |
| core/match AND, frequent + mid | 2.95 | 1.34 | 2.56 | 0.86 | 2.21x |
| core/match OR, 2 frequent | 10.53 | 3.64 | 6.28 | 3.72 | 2.90x |
| core/match OR, 4 mixed | 4.17 | 1.71 | 3.59 | 1.44 | 2.44x |
| core/match, sort by price | 6.07 | 1.04 | 3.17 | 0.89 | 5.83x |
| core/match_all, sort by num | 2.51 | 0.54 | 2.55 | 0.46 | 4.63x |
| core/match_phrase, 2 frequent | 63.95 | 45.77 | 49.29 | 48.03 | 1.40x |
| core/match_phrase, slop 2 | 181.49 | 132.67 | 151.38 | 152.36 | 1.37x |
| core/match_phrase_prefix | 33.35 | 15.76 | 22.86 | 19.77 | 2.12x |
| core/match_phrase_prefix, rare lead | 27.63 | 11.36 | 16.79 | 13.64 | 2.43x |
| core/match_phrase_prefix, wide expansion | 224.54 | 237.48 | 233.86 | 229.87 | 0.95x |
| core/multi_match body+title | 4.50 | 2.49 | 3.44 | 1.84 | 1.80x |
| core/percentiles price | 117.06 | 100.70 | 107.58 | 109.98 | 1.16x |
| core/prefix on title | 3.54 | 1.61 | 3.15 | 1.43 | 2.20x |
| core/range num 4 > avg price | 15.01 | 14.69 | 18.17 | 16.97 | 1.02x |
| core/range on long | 2.35 | 0.85 | 1.97 | 0.38 | 2.76x |
| core/span_near over span_or | 728.58 | 697.65 | 831.64 | 832.72 | 1.04x |
| core/span_near, slop 3 in order | 218.92 | 218.84 | 209.21 | 217.21 | 1.00x |
| core/stats on price | 12.19 | 4.74 | 7.72 | 6.49 | 2.57x |
| core/t0 sort num desc, 10 | 5.31 | 0.72 | 1.46 | 0.33 | 7.41x |
| core/t0 sort tag asc, 100 | 6.82 | 1.19 | 4.36 | 2.78 | 5.73x |
| core/t0, top_hits 10 | 39.99 | 17.08 | 21.32 | 19.23 | 2.34x |
| core/t5 fetch 100 _source | 8.40 | 2.46 | 3.46 | 1.62 | 3.42x |
| core/t5 fetch 100 docvalue_fields | 8.32 | 2.00 | 4.07 | 1.82 | 4.16x |
| core/t8, terms tag > top_hits 3 | 39.36 | 17.50 | 19.55 | 16.78 | 2.25x |
| core/term on keyword | 1.99 | 0.60 | 1.78 | 0.26 | 3.29x |
| core/term, frequent (t0) | 3.42 | 1.78 | 5.93 | 0.96 | 1.92x |
| core/term, mid (t2s) | 2.66 | 0.91 | 3.51 | 0.43 | 2.91x |
| core/term, rare (t1z4) | 2.36 | 0.70 | 3.07 | 0.32 | 3.37x |
| core/terms agg on category | 1.63 | 0.48 | 2.46 | 0.30 | 3.44x |
| core/terms agg on tag, filtered | 10.00 | 3.87 | 5.62 | 3.70 | 2.59x |
| core/terms category > percentiles price | 101.13 | 112.12 | 116.49 | 112.95 | 0.90x |
| core/terms category > stats price | 59.85 | 14.60 | 14.95 | 13.35 | 4.10x |
| core/terms category > top_hits num desc tracked | 67.11 | 40.83 | 48.10 | 50.31 | 1.64x |
| core/terms category, filtered t8 | 9.48 | 2.94 | 4.24 | 2.86 | 3.22x |
| core/terms num (numeric) 25 | 240.74 | 88.58 | 112.80 | 110.35 | 2.72x |
| core/terms tag > terms category | 65.78 | 35.18 | 35.77 | 33.86 | 1.87x |
| core/terms tag > terms num 10 | 333.82 | 130.59 | 186.91 | 183.89 | 2.56x |
| core/terms tag, lone | 1.45 | 0.26 | 1.62 | 0.10 | 5.61x |
| exists/exists price | 1.41 | 0.55 | 1.33 | 0.11 | 2.57x |
| fetch/fetch 10 _source | 2.21 | 0.87 | 1.83 | 0.42 | 2.53x |
| fetch/fetch 10 explain | 2.17 | 2.00 | 1.92 | 0.35 | 1.09x |
| fetch/fetch 100 + inner source (fields) | 8.12 | 3.91 | 4.61 | 1.83 | 2.07x |
| fetch/fetch 100 _source | 5.61 | 2.17 | 3.73 | 1.71 | 2.58x |
| fetch/fetch 100 _source includes 2 fields | 6.30 | 2.19 | 4.01 | 1.84 | 2.88x |
| fetch/fetch 100 stored fields none + docvalue | 6.58 | 2.25 | 3.92 | 1.62 | 2.92x |
| floor/floor match_none | 0.87 | 0.19 | 1.29 | 0.05 | 4.55x |
| floor/floor size0 term count | 0.77 | 0.22 | 1.24 | 0.02 | 3.56x |
| function_score/function_score field_value_factor | 92.82 | 55.34 | 60.82 | 62.72 | 1.68x |
| function_score/function_score gauss decay on price | 97.78 | 55.31 | 60.77 | 62.89 | 1.77x |
| function_score/function_score random_score | 70.35 | 68.51 | 78.02 | 78.72 | 1.03x |
| function_score/function_score weight+filter x3 | 38.09 | 33.95 | 37.75 | 36.94 | 1.12x |
| fuzzy/fuzzy body t0x AUTO | 14.26 | 8.54 | 9.62 | 8.08 | 1.67x |
| fuzzy/fuzzy body t2s fuzziness 1 | 31.89 | 20.37 | 21.59 | 20.64 | 1.57x |
| fuzzy/match fuzziness AUTO 2 terms | 88.75 | 39.32 | 62.00 | 56.59 | 2.26x |
| geo/geo_distance 50km filter | 2.15 | 0.93 | 1.72 | 0.45 | 2.31x |
| geo/geo_distance sort | 74.76 | 55.28 | 60.05 | 64.44 | 1.35x |
| geo/geo_polygon | 2.62 | 1.33 | 1.95 | 0.67 | 1.97x |
| highlight/plain highlight body | 6.15 | 9.95 | 10.40 | 7.62 | 0.62x |
| highlight/unified highlight 100 hits | 10.45 | 7.24 | 11.24 | 3.49 | 1.44x |
| highlight/unified highlight body top 10 | 8.10 | 8.86 | 10.18 | 7.59 | 0.91x |
| highlight/unified phrase highlight | 39.59 | 31.53 | 32.27 | 31.33 | 1.26x |
| ids/ids 20 | 1.41 | 0.71 | 1.72 | 0.28 | 1.98x |
| join/children agg | 18.54 | 22.83 | 25.81 |  | 0.81x |
| join/has_child score_mode sum | 6.99 | 13.56 | 7.91 | 6.57 | 0.52x |
| join/has_child term | 3.41 | 6.03 | 4.63 | 3.48 | 0.57x |
| join/has_parent | 6.74 | 8.28 | 5.93 | 4.68 | 0.81x |
| join/parent agg | 3.50 | 4.27 | 4.51 |  | 0.82x |
| join/parent_id | 1.11 | 0.28 | 1.28 | 0.13 | 3.92x |
| nested/nested agg avg stars | 30.87 | 27.10 | 28.65 | 27.39 | 1.14x |
| nested/nested agg terms > reverse_nested | 57.52 | 54.01 | 53.89 | 56.79 | 1.06x |
| nested/nested bool author+stars | 2.21 | 2.02 | 1.96 | 0.63 | 1.09x |
| nested/nested match text | 17.23 | 17.01 | 14.27 | 13.39 | 1.01x |
| nested/nested score_mode max + inner_hits | 10.63 | 3.63 | 3.26 | 0.87 | 2.93x |
| nested/nested term | 2.12 | 1.81 | 2.00 | 0.42 | 1.17x |
| other-types/boolean term | 1.45 | 1.44 | 1.76 | 0.26 | 1.01x |
| other-types/ip cidr | 1.67 | 1.32 | 1.51 | 0.24 | 1.26x |
| other-types/ip range agg | 14.95 | 12.75 | 16.79 |  | 1.17x |
| other-types/ip term | 1.27 | 1.08 | 1.38 | 0.16 | 1.18x |
| prefix/prefix keyword tag g1 | 1.80 | 0.88 | 2.02 | 0.68 | 2.04x |
| prefix/prefix text title t1z (size 100) | 6.62 | 3.04 | 3.85 | 2.27 | 2.17x |
| query_string/query_string field:term + range | 3.08 | 2.11 | 3.25 | 1.55 | 1.46x |
| query_string/query_string multi-field | 5.10 | 2.48 | 3.75 | 2.37 | 2.06x |
| query_string/query_string phrase | 84.26 | 46.84 | 57.11 | 54.13 | 1.80x |
| query_string/query_string t1 AND t2 | 9.83 | 3.89 | 6.43 | 4.43 | 2.53x |
| query_string/query_string t1 OR t2 -t3 | 35.40 | 12.91 | 16.50 | 13.87 | 2.74x |
| query_string/query_string wildcard+fuzzy | 10.64 | 4.74 | 7.01 | 5.47 | 2.24x |
| query_string/simple_query_string | 36.31 | 28.82 | 40.26 | 38.13 | 1.26x |
| range/bool filter date + term | 2.16 | 1.44 | 2.44 | 0.96 | 1.50x |
| range/range date 1 month | 1.61 | 0.96 | 1.52 | 0.33 | 1.69x |
| range/range double | 1.75 | 0.77 | 1.44 | 0.28 | 2.26x |
| range/range keyword | 1.57 | 0.42 | 1.58 | 0.16 | 3.73x |
| regexp/regexp keyword c[1-3][0-9] | 2.81 | 1.57 | 2.40 | 1.10 | 1.79x |
| regexp/regexp keyword g.5.* | 2.12 | 1.16 | 1.99 | 0.68 | 1.84x |
| regexp/regexp text title t1z[a-c].* | 1.71 | 0.77 | 1.48 | 0.34 | 2.21x |
| rescore/rescore phrase top 50 | 11.99 | 6.36 | 7.42 | 5.83 | 1.89x |
| script/avg agg with script | 39.55 | 22.15 | 33.97 | 30.95 | 1.79x |
| script/script query (filter) | 4.37 | 2.47 | 3.35 | 2.05 | 1.77x |
| script/script_fields x2 | 2.08 | 1.77 | 1.80 | 0.37 | 1.18x |
| script/script_score doc['num'] | 47.08 | 23.84 | 26.80 | 25.10 | 1.97x |
| script/scripted_metric | 31.56 | 26.13 | 29.06 |  | 1.21x |
| script/sort by script | 53.31 | 83.24 | 80.71 | 85.70 | 0.64x |
| script/terms agg with script | 313.38 | 224.16 | 254.13 |  | 1.40x |
| sort/search_after by num | 2.62 | 1.44 | 2.91 | 1.14 | 1.81x |
| sort/size 1000 by score | 43.00 | 20.76 | 24.86 | 16.55 | 2.07x |
| sort/sort _score + _doc tiebreak | 8.62 | 3.69 | 5.74 | 4.35 | 2.33x |
| sort/sort date desc size 100 | 10.16 | 1.92 | 3.36 | 1.36 | 5.28x |
| sort/sort keyword desc + num asc | 3.60 | 1.73 | 2.69 | 1.19 | 2.08x |
| suggest/completion fuzzy | 1.41 | 1.49 | 1.67 | 0.11 | 0.95x |
| suggest/completion prefix | 1.31 | 1.31 | 1.53 | 0.09 | 1.00x |
| suggest/phrase suggester | 0.94 | 0.95 | 1.36 |  | 0.99x |
| suggest/search_as_you_type | 1.94 | 1.62 | 1.98 | 0.38 | 1.20x |
| suggest/term suggester | 0.90 | 0.91 | 0.74 |  | 0.99x |
| terms/terms keyword 20 values | 1.64 | 1.06 | 2.16 | 0.66 | 1.55x |
| terms/terms long 50 values | 2.04 | 1.36 | 2.35 | 0.88 | 1.50x |
| totals/collapse on category | 24.02 | 12.32 | 14.03 | 14.05 | 1.95x |
| totals/post_filter + terms agg | 33.77 | 6.04 | 6.90 | 5.42 | 5.59x |
| totals/track_total_hits false | 7.83 | 3.27 | 4.25 | 3.17 | 2.40x |
| totals/track_total_hits true (default 10k) | 8.54 | 3.43 | 4.94 | 3.61 | 2.49x |
| wildcard/wildcard case-insensitive | 2.20 | 1.44 | 2.24 | 0.90 | 1.53x |
| wildcard/wildcard keyword *5 | 2.09 | 1.24 | 2.12 | 0.65 | 1.69x |
| wildcard/wildcard keyword c?3 | 1.69 | 0.73 | 1.66 | 0.22 | 2.32x |
| wildcard/wildcard keyword g1* | 2.21 | 0.95 | 2.26 | 0.66 | 2.34x |
| wildcard/wildcard text title t1z* | 3.60 | 1.62 | 2.41 | 1.16 | 2.22x |
