# Testing and benchmarking without Docker

Everything that used to start a container now starts a process. A cloud VM, a CI runner and a
WSL2 shell take the same commands, and nothing in the repository needs a container runtime.
`scripts/docker-test.sh` stays as an optional way to cap a build on a machine that has Docker; no
test, e2e proof, bench or CI job calls it.

```
scripts/lr doctor                  what this box has and lacks
scripts/lr build                   native library (dist profile) and plugin zip
scripts/lr test quick|e2e|full|ci  the tiers below
scripts/lr bench opensearch|realtime|tenants|memory|ab|knn|text|conc
scripts/lr corpus synthetic|loghub|bgl|taxi|marco ...
scripts/lr node run|kill|start|stop|rm|logs|list ...
```

## The pieces

**Nodes are tarball distributions** (`opensearch-plugin/e2e/nodectl.py`, standard library only).
OpenSearch ships as `opensearch-3.9.0-linux-{x64,arm64}.tar.gz` with its own JDK 25; nodectl
downloads it once into `$NODE_HOME` (default `~/os-native`), builds *variants* from it and runs
nodes from them:

| variant | what it is | what it replaces |
|---|---|---|
| `stock` | the tarball as shipped (all bundled plugins) | `opensearchproject/opensearch` |
| `stock-min` | no bundled plugins | `opensearch-stock-min` / `lucene-rust-stock` |
| `plugin` | this repo's zip over `opensearch-custom-codecs`, the encrypted-store test key, `--enable-native-access` | `lucene-rust-opensearch` |
| `plugin-knn` | the same, also keeping `opensearch-knn` (upgraded 2.x indices) | the upgrade job's image |

`nodectl.py dist NAME --version V --keep P... --zip Z` makes others (the 2.19 node of the upgrade
proof is `--version 2.19.6 --keep opensearch-knn`). A variant is a copy of the distribution whose
unmodified trees (`jdk`, `lib`, `modules`) are hard links, so it costs a few MB and a rebuilt plugin
zip is picked up by content hash. `run NAME VARIANT PORT` takes `--heap`, `--setting k=v`, `--env`,
`--cpus` (taskset), `--mem-limit` (a memory cgroup with swap off, root only, v1 or v2), `--data`,
`--transport`, and `--lib X.so` to run another build of the native library over the same zip,
which is what the A/B scripts do. `kill`, `start`, `stop`, `restart`, `rm`, `logs`, `pid`, `rss`,
`oom` do what the matching docker commands did; the e2e Python imports the same functions
(`nodectl.kill(name)`, `nodectl.exec_(name, script)`). Root shells run nodes as the `osbench` user,
which OpenSearch requires.

**One bench client.** Every timing script goes through `opensearch-plugin/e2e/benchhttp.py`
(a raw HTTP/1.1 keep-alive socket per thread, time-based warmup, p50/p90/p99). Do not add a
`urllib`, `requests` or `http.client` call to a timed path; `scripts/test-benchhttp.py` checks the
client.

**Corpora** (`benchmarks/corpus/fetch.py`, also `scripts/lr corpus`). Nothing needs a login, and
every fetch is cached:

| corpus | command | size | used by |
|---|---|---|---|
| synthetic text, numerics, geo (seeded Zipf: the same bytes everywhere) | `lr corpus synthetic DIR 1000000` | seconds, no download | `bench-opensearch.sh`, realtime, tenants, memory, `native-node/ab.sh` |
| Loghub bgl/hdfs/openssh/thunderbird lines | `lr corpus loghub DIR --mb 250` | up to 1 GB | newest-n, agg-segment-cache |
| NYC yellow taxi 2025-01/02 parquet | `lr corpus taxi DIR` then `columnar-e2e/prep.py taxi` | 120 MB | columnar-e2e |
| Loghub BGL | `lr corpus bgl DIR` then `columnar-e2e/prep.py bgl` | 700 MB | columnar-e2e |
| MS MARCO passages | `lr corpus marco OUT.jsonl 1000000` (streams the first N, never the 1.1 GB) | N lines | `native-node/text_bench.py prep` |
| clustered random vectors with exact neighbours | `MODE=knn` in `native-node/ab.sh` | generated per run | k-NN A/B |

The synthetic corpus is the default for anything that has to run in CI or in a few minutes; a real
corpus is for numbers that go in a report.

## Tiers

| tier | runs | where | time (4 cores) |
|---|---|---|---|
| `quick` | static checks (arithmetic, port invariants, parity ledger, wire constants, dev notes), `test-nodectl.py`, `test-benchhttp.py`, then `cargo nextest` for the crates changed since `origin/main` (the whole workspace when `Cargo.toml`/`Cargo.lock` changed) | any laptop | minutes |
| `e2e` | `quick` plus one real node: the native-vs-Lucene query matrix, force-merge reader release, SIGKILL recovery (`verify-opensearch.sh --docs 20000`) | a cloud box | about 25 min with a cold build |
| `full` | `scripts/gate.sh` plus: YAML suites against plugin and stock with failure sets compared, the engine proof (`--engine --yaml`), the three-node cluster, rolling upgrade and rollback, 2.19 to 3.9 upgrade, k-NN, hybrid, neural, sparse and pattern-text checks | a cloud box | a few hours |
| `ci` | exactly the OpenSearch jobs of `.github/workflows/ci.yml`, in sequence | a runner, or a box that wants to reproduce one | 1-2 hours |

CI keeps its parallel jobs; each calls the same scripts the tiers do (`verify-opensearch.sh`,
`verify-opensearch-cluster.sh`, `verify-opensearch-upgrade.sh`, `check-knn-*.py`), so a red job
reproduces with `scripts/lr test ci` or by running its script. The runners are free x64 and arm64
GitHub runners; arm64 downloads the arm64 tarball. Merge rules (local checks, no CI wait before the
first release) are unchanged.

## Where each Docker use went

| was | now |
|---|---|
| `verify-opensearch.sh`, `verify-opensearch-cluster.sh`, `verify-rollback.sh`, `verify-opensearch-upgrade.sh` (`docker build/run/stop/logs`, volumes, a holder container for loopback networking) | `nodectl.py run/stop/logs`, plain directories for data, loopback transport on distinct ports (`9301..9303`) |
| `opensearch-plugin/docker/Dockerfile` | deleted: `nodectl.py dist plugin` |
| `scripts/opensearch-dist.sh` (`docker cp` of `lib/` from the image) | copies `lib/` and `modules/lang-painless` from the tarball, registers its JDK 25 as a Gradle toolchain |
| the glibc check of the library against the image | gone: the library runs on the host's own glibc |
| `check-knn-*.py`, `check-hybrid.py`, `check-neural-plugin.py`, `check-ml-plugin.py` (a mount of `libtorch` and a proxy truststore into the container), `check-sparse-plugin.py`, `check-tensor-plugin.py`, `check-pattern-text-plugin.py`, `bench-sparse-plugin.py`, `gen-seismic-fixture.py` | `nodectl.run/rm/logs` with the variants above; a mount is a host path now |
| `check-seismic-writer.sh`, `check-zstd-stored-fields.sh` (Java inside the image) | Java with the tarball's JDK and jars |
| `bench-opensearch.sh`, `bench-realtime.sh`, `bench-tenants.sh`, `analyze-tenants.sh`, `bench-memory-bounds.sh`, `bench-fetch-memory.sh` and the e2e Python behind them (`docker exec`, `docker stats`, `docker top`, `docker kill`) | `scripts/lib-nodes.sh` (`os_up`, `os_down`) and `nodectl.exec_/rss/pid/open_files/kill`; `MEM_LIMIT` is a cgroup instead of `--memory` |
| `benchmarks/columnar-e2e`, `agg-sweep`, `agg-segment-cache`, `newest-n`, `ivf`, `lean_seq_no_bench` | `nodectl.py run ... --lib X.so` per build |
| `scripts/docker-test.sh` as the gate's cap | `scripts/capped.sh` (a cgroup, root or `systemd-run --user`); `gate.sh` itself always ran natively |

Already Docker-free before this work: `soak.sh`, `benchmarks/native-node/*` (which keep their own
launcher, `native-node.sh`, with the same layout under `$NODE_HOME/stock`), and every Rust test.

## A cloud box from scratch

```
git clone https://github.com/huyz0/lucene-rust && cd lucene-rust
scripts/lr doctor               # cargo (rust-toolchain.toml), JDK 21 + Gradle 8.14.3, python3
scripts/lr build                # library (about 5 min) and zip
scripts/lr test e2e
scripts/lr corpus synthetic /data/corpus 1000000
CORPUS=/data/corpus BUILDS="main=/p/main.so,new=/p/new.so" scripts/lr bench ab
```

Keep the driver off the node's cores (`NODE_CPUS=0-2`, driver on core 3), take 100+ samples for
anything under 3 ms, and expect under 10% spread between rounds on a quiet 4-core box; shared
Windows hosts showed 2-3x.

## Limits

- Linux only (x64 and arm64). On Windows use WSL2.
- A node needs 1.1 GB of anonymous memory with a 512 MB heap and about 1.6 GB with 1 GB, so a
  `MEM_LIMIT` below that gets the JVM OOM-killed (`nodectl.py oom NAME` says so).
- Where egress goes through a TLS-intercepting proxy, `check-ml-plugin.py` and `check-neural-plugin.py` need
  `KEEP_JAVA_TOOL_OPTIONS=1`: the stock node's DJL downloads libtorch, and the node gets the proxy and truststore
  `-D` flags of the shell's `JAVA_TOOL_OPTIONS`.
- Nothing is isolated beyond what `--cpus` and `--mem-limit` give: ports `9200-9203`, `9240-9262`,
  `9270` and `19200+` are used by the scripts, one run at a time per box.
- The tarball is the release artifact, not the Docker image: `config/` differs slightly (security
  plugin handling, `cluster.name`), which nodectl sets explicitly.
