#!/usr/bin/env python3
"""Faiss baselines for the IVF benchmark, on the files prepare.py writes.

    faiss_bench.py <prefix> <l2|cosine> [--nlist N] [--queries N] [--nprobe 8,16,...]
                   [--rerank 0,100,...] [--indexes flat,sq8,rabitq,pq] [--batched]

Builds with every core and searches on one thread (omp_set_num_threads(1)),
one query per call, so QPS is single-thread QPS, as in ivf_bench. --batched
passes all queries in one call instead, which lets Faiss rank centroids for
the whole batch with one matrix multiply; that is faster per query than any
one-query-at-a-time engine can be and is not what ivf_bench measures. Prints the same CSV columns:
index,config,build_s,recall@10,latency_us,qps,pinned_mb,vectors_mb.
`pinned_mb` is the serialized index without its refine store (centroids and
codes); `vectors_mb` is the full-precision vectors a rerank reads.

Indexes (all over the same nlist):
  flat    IVF,Flat: exact distances in the probed clusters (the ceiling)
  sq8     IVF,SQ8, reranked with IndexRefineFlat when rerank > 0
  rabitq  IVF,RaBitQ (1-bit codes, 4-bit queries via qb), reranked likewise
  pq      IVF,PQ(dim/2)x4fs fast-scan, reranked likewise
"""
import argparse
import time

import faiss
import numpy as np

K = 10


def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        return np.fromfile(f, dtype=dtype, count=int(n) * int(d)).reshape(int(n), int(d))


def recall(found, gt):
    hits = 0
    for f, g in zip(found, gt):
        hits += len(set(f[:K].tolist()) & set(g[:K].tolist()))
    return hits / (len(gt) * K)


def size_mb(index):
    return faiss.serialize_index(index).nbytes / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix")
    p.add_argument("metric")
    p.add_argument("--nlist", type=int, default=0)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--nprobe", default="4,8,16,32,64,128")
    p.add_argument("--rerank", default="0,50,100")
    p.add_argument("--indexes", default="flat,sq8,rabitq,pq")
    p.add_argument("--batched", action="store_true")
    a = p.parse_args()
    base = read_bin(a.prefix + ".base.fbin", np.float32)
    queries = read_bin(a.prefix + ".query.fbin", np.float32)[: a.queries]
    gt = read_bin(a.prefix + ".gt.ibin", np.int32)[: a.queries]
    n, dim = base.shape
    metric = faiss.METRIC_L2
    if a.metric == "cosine":
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        metric = faiss.METRIC_INNER_PRODUCT
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    nlist = a.nlist or max(1, n // 384)
    threads = faiss.omp_get_max_threads()
    print("index,config,build_s,recall@10,latency_us,qps,pinned_mb,vectors_mb")
    vectors_mb = base.nbytes / 1e6
    specs = {
        "flat": f"IVF{nlist},Flat",
        "sq8": f"IVF{nlist},SQ8",
        "rabitq": f"IVF{nlist},RaBitQ",
        "pq": f"IVF{nlist},PQ{dim // 2}x4fs",
    }
    for name in a.indexes.split(","):
        faiss.omp_set_num_threads(threads)
        t = time.time()
        index = faiss.index_factory(dim, specs[name], metric)
        index.train(base)
        index.add(base)
        build = time.time() - t
        pinned = size_mb(index)
        refine = None
        if name != "flat":
            # IndexRefineFlat keeps its own copy of the vectors.
            refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(base))
        faiss.omp_set_num_threads(1)
        for nprobe in [int(x) for x in a.nprobe.split(",")]:
            reranks = [0] if name == "flat" else [int(x) for x in a.rerank.split(",")]
            for rerank in reranks:
                if rerank == 0 or refine is None:
                    params = faiss.SearchParametersIVF(nprobe=nprobe)
                    if name == "rabitq":
                        params = faiss.IVFRaBitQSearchParameters(nprobe=nprobe, qb=4)
                    target = index
                else:
                    inner = faiss.SearchParametersIVF(nprobe=nprobe)
                    if name == "rabitq":
                        inner = faiss.IVFRaBitQSearchParameters(nprobe=nprobe, qb=4)
                    params = faiss.IndexRefineSearchParameters(
                        k_factor=rerank / K, base_index_params=inner
                    )
                    target = refine
                t = time.time()
                if a.batched:
                    _, found = target.search(queries, K, params=params)
                else:
                    found = np.vstack(
                        [target.search(queries[i : i + 1], K, params=params)[1] for i in range(len(queries))]
                    )
                secs = time.time() - t
                cfg = f"nlist={nlist} nprobe={nprobe} rerank={rerank}"
                print(
                    f"faiss-{name},{cfg},{build:.1f},{recall(found, gt):.4f},"
                    f"{secs / len(queries) * 1e6:.0f},{len(queries) / secs:.0f},"
                    f"{pinned:.1f},{vectors_mb:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
