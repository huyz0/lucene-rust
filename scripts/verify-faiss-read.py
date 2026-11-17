#!/usr/bin/env python3
"""Opens indexes crates/lucene-faiss wrote (example write_for_faiss) with the real
Faiss library and checks that Faiss searches them with high recall against its
own exact search over the same vectors.

  cargo run -q -p lucene-faiss --example write_for_faiss DIR
  /tmp/fv/bin/python scripts/verify-faiss-read.py DIR
"""
import os, sys
import numpy as np
import faiss

D = 24
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "data", "faiss")
x = np.fromfile(os.path.join(ROOT, "hnsw_idmap_l2.vectors"), dtype="float32").reshape(-1, D)
q = np.fromfile(os.path.join(ROOT, "hnsw_idmap_l2.queries"), dtype="float32").reshape(-1, D)
ids = (np.arange(len(x)) * 3 + 7).astype("int64")
out = sys.argv[1]
ok = True
for name, metric in [("ours_hnsw_l2", faiss.METRIC_L2), ("ours_hnsw_ip", faiss.METRIC_INNER_PRODUCT),
                     ("ours_ivf_l2", faiss.METRIC_L2), ("ours_ivf_ip", faiss.METRIC_INNER_PRODUCT)]:
    idx = faiss.read_index(os.path.join(out, name + ".idx"))
    exact = faiss.IndexIDMap(faiss.IndexFlat(D, metric))
    exact.add_with_ids(x, ids)
    _, want = exact.search(q, 10)
    ps = faiss.ParameterSpace()
    ps.set_index_parameters(idx, "efSearch=64" if "hnsw" in name else "nprobe=8")
    _, got = idx.search(q, 10)
    recall = np.mean([len(set(g) & set(w)) / 10 for g, w in zip(got, want)])
    floor = 0.95 if "hnsw" in name else 0.8
    print(f"{name}: faiss {faiss.__version__} reads it, ntotal={idx.ntotal}, recall@10={recall:.3f}")
    ok &= recall >= floor
sys.exit(0 if ok else 1)
