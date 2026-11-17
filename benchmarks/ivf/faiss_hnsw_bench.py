#!/usr/bin/env python3
"""Faiss HNSW baseline on prepare.py's files: faiss_hnsw_bench.py <prefix> <l2|cosine> [--queries N].
Single-thread one-query-per-call search; same CSV columns as ivf_bench."""
import argparse, time
import faiss, numpy as np

def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        return np.fromfile(f, dtype=dtype, count=int(n) * int(d)).reshape(int(n), int(d))

p = argparse.ArgumentParser()
p.add_argument("prefix"); p.add_argument("metric")
p.add_argument("--queries", type=int, default=1000)
p.add_argument("--ef", default="16,32,64,128,256")
a = p.parse_args()
base = read_bin(a.prefix + ".base.fbin", np.float32)
q = read_bin(a.prefix + ".query.fbin", np.float32)[: a.queries]
gt = read_bin(a.prefix + ".gt.ibin", np.int32)[: a.queries]
metric = faiss.METRIC_L2
if a.metric == "cosine":
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    metric = faiss.METRIC_INNER_PRODUCT
base = np.ascontiguousarray(base, dtype=np.float32); q = np.ascontiguousarray(q, dtype=np.float32)
n, d = base.shape
print("index,config,build_s,recall@10,latency_us,qps,pinned_mb,vectors_mb")
t = time.time()
idx = faiss.IndexHNSWFlat(d, 16, metric)
idx.hnsw.efConstruction = 100
idx.add(base)
build = time.time() - t
mb = faiss.serialize_index(idx).nbytes / 1e6
faiss.omp_set_num_threads(1)
for ef in [int(x) for x in a.ef.split(",")]:
    idx.hnsw.efSearch = ef
    t = time.time()
    found = np.vstack([idx.search(q[i:i+1], 10)[1] for i in range(len(q))])
    s = time.time() - t
    hits = sum(len(set(f.tolist()) & set(g[:10].tolist())) for f, g in zip(found, gt))
    print(f"faiss-hnsw,M=16 efC=100 ef={ef},{build:.1f},{hits/(len(q)*10):.4f},{s/len(q)*1e6:.0f},{len(q)/s:.0f},{mb:.1f},{n*d*4/1e6:.1f}", flush=True)
