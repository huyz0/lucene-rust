# Native-node benches (no Docker)

Runs stock OpenSearch 3.9.0 and the plugin as plain processes, for hosts without a container
runtime (cloud sandboxes, CI boxes) and for A/B runs that need a quiet machine. Tested on a
4-core / 15 GB Linux box as root: the node gets cores 0-2, the driver core 3.

```
source benchmarks/native-node/native-node.sh
NATIVE_LIB=target/dist/liblucene_ffi.so native_setup   # tarball, plugin zip (Gradle), install
cargo build --profile dist -p lucene-ffi --features mimalloc   # the library to measure (about 5 min)
python3 opensearch-plugin/e2e/bench_nodes.py gen /data/corpus 1000000
CORPUS=/data/corpus BUILDS="main=/p/main.so,new=/p/new.so" benchmarks/native-node/ab.sh
```

`ab.sh` loads one force-merged index per side, then alternates stock and each build for
`ROUNDS_AB` rounds (restarting the node each time) and prints the best median per shape.
`MODE=knn` (one persistent socket, since urllib's connection per request costs more than a search) times `lucene`-engine HNSW k-NN instead (`NVEC` clustered random vectors of `DIM`, `DATA_DIR` apart from the search one).
`MEM_LIMIT=2G` puts the node in a memory cgroup (cgroup v1, needs root): the page cache counts
against it, so it is real memory pressure. `NODE_CPUS`, `HEAP`, `PORT`, `NODE_ARGS`, `SHAPES`
(substrings), `ROUNDS`/`WARMUP` (requests per shape) are also read.

What the script handles: OpenSearch refuses to run as root (a `osbench` user is created);
the bundled JDK 25 is used (`JAVA_HOME` and the sandbox's `JAVA_TOOL_OPTIONS` are unset for the
node); bundled plugins other than `opensearch-custom-codecs` are removed from the plugin node, as
in `opensearch-plugin/e2e/nodectl.py`, variant `plugin` (neural-search's query phase searcher conflicts); the
library inside the zip must be named `liblucene_ffi.so` (`NATIVE_LIB` copies it under that name);
`bench_nodes.py run` takes `native` as its CONTAINER argument to read the node's RSS with `ps`.

Run-to-run spread on the cloud box is under 10% between rounds (three alternating rounds agree),
against the 2-3x of the shared Windows host.

Node memory needs: stock 3.9.0 uses about 1.1 GB of anonymous memory with a 512 MB heap and about 1.6 GB with 1 GB, so a cgroup below that gets the JVM OOM-killed; the page cache gets whatever is left.

## Ranked text search on a real corpus

`text_bench.py` and `text-ab.sh` time BM25 top-k search (term, match OR/AND, `minimum_should_match`, phrase, bool with filters, real MS MARCO dev queries, size 100, exact totals) over MS MARCO passages, stock against plugin builds, with the node's native query and fetch time from the plugin stats. `prof.sh` runs a warmed closed loop per shape under async-profiler (`cpu` event) and writes collapsed stacks and node CPU per request.

```
QUERIES_TSV=queries.dev.tsv python3 benchmarks/native-node/text_bench.py prep marco1m.jsonl /data/marco 1000000
OUT_DIR=/data/marco BUILDS="main=/p/main.so,new=/p/new.so" benchmarks/native-node/text-ab.sh
OUT_DIR=/data/marco PROF_EVENT=cpu AP=/path/asprof PROF_SHAPES="term frequent|real queries (OR)" benchmarks/native-node/prof.sh plugin /p/main.so
```

The node runs as `osbench`, so the path to the node home and the libraries must be traversable by it (`chmod o+x /root`), and async-profiler writes through `/tmp/prof` (mode 777). The `itimer` event crashed JDK 25 here; `cpu` works.

## Long documents, several shards, concurrent clients

`wiki_bench.py` + `wiki-ab.sh`: English Wikipedia (HF `wikimedia/wikipedia` parquet; needs `pyarrow`), `title` + `text`, `SHARDS` shards (default 4), 25 query shapes built from the corpus (term bands, phrases, 6-word fragments, titles, multi_match, filters, size 100, exact totals). `wiki-ab.sh` alternates stock and plugin builds; per node it runs the serial timing and then closed-loop concurrent load (`SHAPES_CC="a|b"`, `CLIENTS="1 8 32"`) printing qps, p50, p99 and node CPU per request. `NOMERGE=1 SETTINGS='{"refresh_interval":"1s"}'` keeps the merge policy's segments instead of force-merging. `wiki_bench.py dump` writes top-10 ids/scores/totals for comparing engines.

```
python3 -I benchmarks/native-node/wiki_bench.py prep wiki.parquet /data/wiki
OUT_DIR=/data/wiki BUILDS="main=/p/main.so" SHAPES_CC="term frequent|fragments 6 words (OR)" benchmarks/native-node/wiki-ab.sh
```

## Throughput and cold start

`cc.sh KIND[:LIB.so] ...` runs `conc.py` per shape (`SHAPES_CC="a|b"`, `OUT_DIR`): closed-loop keep-alive raw-socket clients (`PROCS` x `CONNS`, default 2 x 4; `PROCS=1 CONNS=1` is serial) and prints qps and node CPU per request. The node takes cores 0-1, the driver 2-3. Use it instead of `text_bench.py time` when shapes are under ~3 ms: that driver's python `http.client` costs about 1.7 ms per request on both engines and 30-request samples are JIT warm-up. `warmcurve.py` prints serial latency per block of requests right after a node start.

## Client rules (every bench)

A search on the fast paths takes 0.3-3 ms, and a client that costs 1-4 ms per request hides the engine difference (both sides pay it). Every timing script goes through `opensearch-plugin/e2e/benchhttp.py`: one raw HTTP/1.1 keep-alive socket per thread (about 0.06-0.1 ms per request, against about 1 ms for `http.client` and about 3.5 ms for urllib's connection per request), `bh.measure` / `bh.warm` (warm for a minimum number of requests and a minimum of seconds, since a freshly started JVM is still compiling after 30 requests), and `bh.summarize` (n, min, p50, p90, p99, mean, max). `bench_nodes.req`, `verify_opensearch.req` and the scripts that had their own urllib `req` delegate to it; do not add a new `urllib`/`requests`/`http.client` call to a timed path. Pin the driver to cores the node does not use (`DRIVER_CPU`), take 100+ samples for anything under 3 ms, and use `conc.py`/`cc.sh` for concurrent throughput (node CPU per request included). `scripts/test-benchhttp.py` checks the client against a local server.

## Feature-family sweep

`families.py` times about 130 shapes that `ab.py` does not (wildcard, regexp, fuzzy, `query_string`, `function_score`, Painless scripts in every context, sort and paging, totals, fetch and highlight, suggesters, every aggregation family, nested and join queries, ip/boolean fields, geo extras), on `bench` (1M docs), `fam` (nested comments, ip, boolean; N docs) and `jn` (parent/child join). `famsweep.sh` loads those indices on stock and on one plugin build, then alternates `ab.py time` and `families.py time` for `ROUNDS_AB` rounds (`FAMILIES` and `SHAPES` narrow what runs; `significant_text` alone takes 13 s a request, so leave it out of quick runs). `families.py report` prints stock and plugin p50/p99 per shape, the ratio, and a per-family geometric mean. The accelerated engine writes neither term vectors nor `completion` fields, so the suggesters run on an `sg` index left on the standard engine on both nodes.

## Where a request's time goes

`split.py BASE LABEL OUT [N]` runs every shape of `ab.py` and `families.py` (plus two floor shapes: `match_none`, a size-0 term count) serially on one keep-alive connection and records per shape the round-trip p50, the node's CPU per request (`/proc`), and, on the plugin, the native query-phase and fetch time per request, the number of requests that fell back to Java and why, the disk read volume and major page faults per request. The Java request path is the round trip minus native and fetch time; `SHAPES=a,b` selects by substring of `family/name`, and a shape slower than 50 ms gets fewer samples (about 3 s each). `loop.py BASE SHAPE SECS [CLIENTS]` is a closed loop of one shape for a profiler. `direct_check.py BASE` compares the direct-search path with OpenSearch's own on a node started with `NODE_ARGS="-E search.accelerated.direct.enabled=true"`: equal ids, scores, totals and sources, and which requests were answered directly.

A bare `GET /` costs about 0.3 ms through this client and a one-shard search about 0.9-1.0 ms on both engines, so cheap shapes are bound by the request path, not the engine. An async-profiler run of that path (`-e cpu`, `-t`) shows no hot spot: netty, REST parsing, the search action, shard context, the query phase's pool hand-off, the fetch phase and the response objects each take 5-15%.
