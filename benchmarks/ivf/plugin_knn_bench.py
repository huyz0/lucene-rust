#!/usr/bin/env python3
"""knn_vector through the lucene-rust plugin on one OpenSearch node: HNSW
(`lucene` engine) against IVF (`lucene_rust` engine, `method: ivf`).

For each mapping the base vectors are bulk-loaded (refresh off, one shard),
refreshed, optionally force-merged to one segment, and then searched one
query at a time over REST. Prints one CSV row per mapping and setting: load
time and rate, merge time, store size, segments, recall@10 against the
dataset's ground truth, and mean and p99 latency as the client sees it
(`took` is printed too, the shard-side time).

  benchmarks/ivf/plugin_knn_bench.py <prefix> <l2|cosine> [--merge] [--queries N]
      [--mappings hnsw,ivf,ivf-f32] [--nprobes 0,32] [--keep]

`<prefix>` names `.base.fbin`, `.query.fbin` and `.gt.ibin` as
`benchmarks/ivf/prepare.py` writes them. Needs Docker and the plugin zip in
opensearch-plugin/build/distributions (`gradle -p opensearch-plugin bundlePlugin`).
"""
import glob, importlib.util, json, os, shutil, struct, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("check_knn", os.path.join(ROOT, "scripts", "check-knn-plugin.py"))
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

NODE = ("knn-bench", 19231)
K = 10
WARMUP = 200
BATCH = 2000


def read_bin(path, fmt):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        data = f.read()
    return n, d, struct.unpack(f"<{n * d}{fmt}", data)


def mapping(name, dim, space):
    if name == "hnsw":
        method = {"name": "hnsw", "engine": "lucene", "space_type": space,
                  "parameters": {"m": 16, "ef_construction": 100}}
    else:
        method = {"name": "ivf", "engine": "lucene_rust", "space_type": space,
                  "parameters": {"target_cluster_size": 384,
                                 "rerank_encoding": "float" if name == "ivf-f32" else "half_float"}}
    return {"type": "knn_vector", "dimension": dim, "method": method}


def timed(port, method, path, body=None, ndjson=False):
    data = None if body is None else (body.encode() if ndjson else json.dumps(body).encode())
    r = urllib.request.Request(f"http://localhost:{port}{path}", data=data, method=method,
                               headers={"Content-Type": "application/x-ndjson" if ndjson else "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(r, timeout=3600) as resp:
        out = json.loads(resp.read() or b"null")
    return out, time.perf_counter() - t


def main():
    args = sys.argv[1:]
    prefix, metric = args[0], args[1]
    opt = lambda name, default: args[args.index(name) + 1] if name in args else default
    space = {"l2": "l2", "cosine": "cosinesimil"}[metric]
    merge = "--merge" in args
    names = opt("--mappings", "hnsw,ivf").split(",")
    nprobes = [int(x) for x in opt("--nprobes", "0,32").split(",")]
    n, dim, base = read_bin(f"{prefix}.base.fbin", "f")
    nq_all, _, queries = read_bin(f"{prefix}.query.fbin", "f")
    _, gt_k, gt = read_bin(f"{prefix}.gt.ibin", "i")
    nq = min(int(opt("--queries", "1000")), nq_all - WARMUP)

    zips = glob.glob(os.path.join(ROOT, "opensearch-plugin", "build", "distributions", "lucene-rust-*.zip"))
    if not zips:
        sys.exit("no plugin zip: run gradle -p opensearch-plugin bundlePlugin")
    docker_dir = os.path.join(ROOT, "opensearch-plugin", "docker")
    for z in glob.glob(os.path.join(docker_dir, "*.zip")):
        os.remove(z)
    shutil.copy(zips[0], docker_dir)
    image = f"lucene-rust-opensearch:{ck.VERSION}"
    subprocess.run(["docker", "build", "-q", "-t", image, docker_dir], check=True, capture_output=True)
    subprocess.run(["docker", "rm", "-f", NODE[0]], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", NODE[0], "-p", f"{NODE[1]}:9200",
                    "-e", "discovery.type=single-node", "-e", "DISABLE_SECURITY_PLUGIN=true",
                    "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", "OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g",
                    "-e", "cluster.routing.allocation.disk.threshold_enabled=false", image],
                   check=True, capture_output=True)
    port = NODE[1]
    print(f"# {prefix}: {n} x {dim}, {nq} queries, merge {merge}", file=sys.stderr)
    print("mapping,setting,load_s,docs_per_s,merge_s,store_mb,segments,recall@10,mean_ms,p99_ms,took_mean_ms")
    try:
        ck.wait(port)
        for name in names:
            index = f"bench-{name}"
            ck.req(port, "PUT", f"/{index}", {
                "settings": {"index.knn": True, "index.lucene_rust.engine": True, "number_of_shards": 1,
                             "number_of_replicas": 0, "refresh_interval": -1},
                "mappings": {"properties": {"v": mapping(name, dim, space)}}})
            t = time.perf_counter()
            for start in range(0, n, BATCH):
                lines = []
                for i in range(start, min(start + BATCH, n)):
                    lines.append('{"index":{"_id":"%d"}}' % i)
                    lines.append('{"v":[%s]}' % ",".join("%.6g" % x for x in base[i * dim:(i + 1) * dim]))
                r, _ = timed(port, "POST", f"/{index}/_bulk", "\n".join(lines) + "\n", ndjson=True)
                assert not r["errors"], json.dumps(r)[:1000]
            timed(port, "POST", f"/{index}/_refresh")
            load = time.perf_counter() - t
            merge_s = 0.0
            if merge:
                _, merge_s = timed(port, "POST", f"/{index}/_forcemerge?max_num_segments=1")
                timed(port, "POST", f"/{index}/_refresh")
            stats = ck.req(port, "GET", f"/{index}/_stats/store,segments")["indices"][index]["primaries"]
            store = stats["store"]["size_in_bytes"] / 1e6
            segments = stats["segments"]["count"]
            for nprobe in (nprobes if name != "hnsw" else [None]):
                def body(q):
                    knn = {"vector": list(queries[q * dim:(q + 1) * dim]), "k": K}
                    if nprobe:
                        knn["method_parameters"] = {"nprobes": nprobe}
                    return {"size": K, "_source": False, "query": {"knn": {"v": knn}}}
                for q in range(nq, nq + WARMUP):
                    timed(port, "POST", f"/{index}/_search", body(q))
                times, took, found = [], [], 0
                for q in range(nq):
                    r, s = timed(port, "POST", f"/{index}/_search", body(q))
                    times.append(s * 1e3)
                    took.append(r["took"])
                    truth = set(gt[q * gt_k:q * gt_k + K])
                    found += sum(1 for h in r["hits"]["hits"] if int(h["_id"]) in truth)
                times.sort()
                setting = "ef=100" if name == "hnsw" else f"nprobes={nprobe}"
                print(f"{name},{setting},{load:.1f},{n / load:.0f},{merge_s:.1f},{store:.1f},{segments},"
                      f"{found / (nq * K):.4f},{sum(times) / nq:.2f},{times[min(nq - 1, nq * 99 // 100)]:.2f},"
                      f"{sum(took) / nq:.2f}", flush=True)
            if "--keep" not in args:
                ck.req(port, "DELETE", f"/{index}")
    finally:
        if "--keep" not in args:
            subprocess.run(["docker", "rm", "-f", NODE[0]], capture_output=True)


if __name__ == "__main__":
    main()
