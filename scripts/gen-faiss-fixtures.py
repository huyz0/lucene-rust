#!/usr/bin/env python3
"""Writes the Faiss index fixtures crates/lucene-faiss reads: indexes made by the
real Faiss library (python `faiss-cpu`, NOT linked into any build), plus the
top-k results Faiss itself returns for a fixed query set, with and without an
id filter.

  python3 -m venv /tmp/fv && /tmp/fv/bin/pip install faiss-cpu numpy
  /tmp/fv/bin/python scripts/gen-faiss-fixtures.py   # -> fixtures/data/faiss/

Per case NAME it writes NAME.idx (faiss.write_index), NAME.vectors (n x d f32
LE, the vectors as added), NAME.queries (q x d f32 LE) and NAME.expected (TSV:
`plain|filtered`, query number, then k `id:distance` pairs), and manifest.tsv:
`NAME  factory  metric  d  n  k  search-parameter`.
"""
import os, struct, sys
import numpy as np
import faiss

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "data", "faiss")
D, N, Q, K = 24, 1500, 20, 10
SEED = 20261117

CASES = [
    # name, factory, metric, params applied at index time, search parameter
    ("hnsw_idmap_l2", "IDMap,HNSW8", faiss.METRIC_L2, "efConstruction=40", "efSearch=48"),
    ("hnsw_idmap_ip", "IDMap,HNSW8", faiss.METRIC_INNER_PRODUCT, "efConstruction=40", "efSearch=48"),
    ("hnsw_plain_l2", "HNSW8", faiss.METRIC_L2, "efConstruction=40", "efSearch=48"),
    ("ivf_idmap_l2", "IDMap,IVF16,Flat", faiss.METRIC_L2, "", "nprobe=6"),
    ("ivf_idmap_ip", "IDMap,IVF16,Flat", faiss.METRIC_INNER_PRODUCT, "", "nprobe=6"),
    ("flat_idmap_l2", "IDMap,Flat", faiss.METRIC_L2, "", ""),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((N, D)).astype("float32")
    q = rng.standard_normal((Q, D)).astype("float32")
    # ids are not ordinals, as a segment's doc ids are not.
    ids = (np.arange(N) * 3 + 7).astype("int64")
    allowed = ids[ids % 2 == 0]
    manifest = []
    for name, factory, metric, build_params, search_param in CASES:
        index = faiss.index_factory(D, factory, metric)
        ps = faiss.ParameterSpace()
        if build_params:
            ps.set_index_parameters(index, build_params)
        if not index.is_trained:
            index.train(x)
        if factory.startswith("IDMap"):
            index.add_with_ids(x, ids)
        else:
            index.add(x)
        faiss.write_index(index, os.path.join(OUT, name + ".idx"))
        if search_param:
            ps.set_index_parameters(index, search_param)
        with open(os.path.join(OUT, name + ".expected"), "w") as f:
            for kind in ("plain", "filtered"):
                if kind == "filtered" and not factory.startswith("IDMap"):
                    continue
                params = None
                if kind == "filtered":
                    sel = faiss.IDSelectorBatch(allowed)
                    params = faiss.SearchParameters(sel=sel)
                    if factory.startswith("IDMap,HNSW"):
                        params = faiss.SearchParametersHNSW(sel=sel, efSearch=48)
                    elif "IVF" in factory:
                        params = faiss.SearchParametersIVF(sel=sel, nprobe=6)
                dist, lab = index.search(q, K, params=params)
                for qi in range(Q):
                    cells = [f"{int(l)}:{struct.unpack('<I', struct.pack('<f', d))[0]:08x}" for l, d in zip(lab[qi], dist[qi])]
                    f.write("\t".join([kind, str(qi)] + cells) + "\n")
        x.tofile(os.path.join(OUT, name + ".vectors"))
        q.tofile(os.path.join(OUT, name + ".queries"))
        manifest.append("\t".join([name, factory, "l2" if metric == faiss.METRIC_L2 else "ip", str(D), str(N), str(K), search_param]))
    np.array(ids).tofile(os.path.join(OUT, "ids.i64"))
    with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print("faiss", faiss.__version__, "->", OUT)


if __name__ == "__main__":
    sys.exit(main())
