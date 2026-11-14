# columnar-e2e

Stock OpenSearch 3.9.0 against the plugin on real data, one list of aggregation shapes, responses
compared. Before/after harness for the whole-document columnar stages.

```
python3 -I prep.py taxi PARQUET_DIR /data/taxi-corpus 0     # NYC yellow taxi 2025-01/02 parquet (0 = all rows)
python3 -I prep.py bgl  bgl.txt     /data/bgl-corpus  0     # Loghub BGL
DATA=taxi CORPUS=/data/taxi-corpus PLUGIN_SO=target/dist/liblucene_ffi.so ./run.sh
```

Sources: `koorukuroo/yellow_tripdata` (`yellow_tripdata_2025-01.parquet`, `-02`) and
`bolu61/loghub_2` (`data/bgl.txt`) on Hugging Face. `prep.py` needs pyarrow for taxi.

`run.sh` loads one index per node (1 shard, no replica), force-merges to one segment, times each
shape over all documents and over the first 10% of the time span (request cache off, 5 warmup then
20 rounds), then prints store size, indexing time and a table of stock ms, plugin ms, ratio and
whether the responses are equal. `exact` is byte-identical JSON, `approx` differs only in float
rounding (1e-9), `sketch` is a percentiles t-digest within 2% of the p1-p99 spread, `DIFF` fails the run (exit 1).

Knobs: `STOCK_SETTINGS` / `PLUGIN_SETTINGS` (index settings JSON per node; the plugin default is
`{"index.engine.accelerated":true}`), `MAPPING` (a mappings JSON file replacing the dataset's),
`TAG` (output name), `SKIP_STOCK=1` (reuse `out/$TAG-stock.json`, for an A/B of two plugin
builds or settings). Nodes get cores 0-2 and the client core 3; a plugin build is a
`--profile dist --features mimalloc` `liblucene_ffi.so` mounted over the image's.
Needs the images `opensearchproject/opensearch:3.9.0` and `lucene-rust-opensearch:3.9.0`
(`scripts/bench-opensearch.sh` builds the latter).

About 2 minutes per dataset at 1M documents, a few more at the full 7.0M (taxi) / 4.6M (bgl).

`survey.py` is the field survey (presence and multi-valued rate of numeric fields in a corpus
sample); `RESULTS.md` holds the stage 4-6 numbers from this harness and their downsides.
