#!/usr/bin/env python3
"""Turns a gensim-data word-vector archive (word2vec text format, gzipped)
into the base/query/ground-truth files the IVF benchmarks read.

    prepare.py <archive.gz> <out-prefix> <metric: l2|cosine> [queries=10000] [k=100]

Writes <out-prefix>.base.fbin and .query.fbin (u32 count, u32 dim, f32 LE
rows) and .gt.ibin (u32 count, u32 k, i32 LE ids of the exact k nearest base
vectors per query, by the metric; cosine compares unit-normalized vectors).
Queries are a seeded random sample of the vectors, removed from the base.
Exact neighbours come from faiss.IndexFlat.
"""
import gzip
import sys

import faiss
import numpy as np


def read_word2vec(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        n, dim = map(int, f.readline().split())
        out = np.empty((n, dim), dtype=np.float32)
        i = 0
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < dim + 1:
                continue
            out[i] = np.asarray(parts[-dim:], dtype=np.float32)
            i += 1
        return out[:i]


def write_bin(path, a, dtype):
    a = np.ascontiguousarray(a, dtype=dtype)
    with open(path, "wb") as f:
        np.asarray(a.shape, dtype=np.uint32).tofile(f)
        a.tofile(f)


def main():
    src, prefix, metric = sys.argv[1:4]
    nq = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    x = read_word2vec(src)
    # Drop zero and duplicate rows, which make exact ties.
    x = x[np.linalg.norm(x, axis=1) > 0]
    x = np.unique(x, axis=0)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(x))
    queries, base = x[perm[:nq]], x[perm[nq:]]
    if metric == "cosine":
        bn = base / np.linalg.norm(base, axis=1, keepdims=True)
        qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(base.shape[1])
        index.add(bn)
        _, gt = index.search(qn, k)
    else:
        index = faiss.IndexFlatL2(base.shape[1])
        index.add(base)
        _, gt = index.search(queries, k)
    write_bin(prefix + ".base.fbin", base, np.float32)
    write_bin(prefix + ".query.fbin", queries, np.float32)
    write_bin(prefix + ".gt.ibin", gt, np.int32)
    print(f"{prefix}: base {base.shape}, queries {queries.shape}, k {k}, {metric}")


if __name__ == "__main__":
    main()
