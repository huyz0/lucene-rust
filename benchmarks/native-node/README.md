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
