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
`MODE=knn` times `lucene`-engine HNSW k-NN instead (`NVEC` clustered random vectors of `DIM`, `DATA_DIR` apart from the search one).
`MEM_LIMIT=2G` puts the node in a memory cgroup (cgroup v1, needs root): the page cache counts
against it, so it is real memory pressure. `NODE_CPUS`, `HEAP`, `PORT`, `NODE_ARGS`, `SHAPES`
(substrings), `ROUNDS`/`WARMUP` (requests per shape) are also read.

What the script handles: OpenSearch refuses to run as root (a `osbench` user is created);
the bundled JDK 25 is used (`JAVA_HOME` and the sandbox's `JAVA_TOOL_OPTIONS` are unset for the
node); bundled plugins other than `opensearch-custom-codecs` are removed from the plugin node, as
in `opensearch-plugin/docker/Dockerfile` (neural-search's query phase searcher conflicts); the
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

## Throughput and cold start

`cc.sh KIND[:LIB.so] ...` runs `conc.py` per shape (`SHAPES_CC="a|b"`, `OUT_DIR`): closed-loop keep-alive raw-socket clients (`PROCS` x `CONNS`, default 2 x 4; `PROCS=1 CONNS=1` is serial) and prints qps and node CPU per request. The node takes cores 0-1, the driver 2-3. Use it instead of `text_bench.py time` when shapes are under ~3 ms: that driver's python `http.client` costs about 1.7 ms per request on both engines and 30-request samples are JIT warm-up. `warmcurve.py` prints serial latency per block of requests right after a node start.
