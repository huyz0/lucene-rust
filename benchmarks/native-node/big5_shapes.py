"""Shape set for wiki_bench.py (BENCH_EXT=big5_shapes): the OpenSearch Benchmark `big5` workload's search operations over the first
N documents of its corpus, so the numbers sit beside published big5 results (1 shard, best_compression, the workload's own mapping).

  python3 -I big5_shapes.py fetch DIR [COMPRESSED_BYTES]    range-request the head of documents-100.json.zst from the workload's CDN
  BENCH_EXT=big5_shapes python3 -I wiki_bench.py prep DIR DIR N_DOCS   bulk files (auto ids) + queries.json from DIR/{index.json,ops.json}
  load with SHARDS=1 SETTINGS='{"codec": "best_compression"...}' (wiki-ab.sh); index.json and operations/default.json come from the workload repo
"""
import json
import os
import re
import sys

INDEX = "big5"
BASE = "https://dbyiw3u3rf9yr.cloudfront.net/corpora/big5/documents-100.json.zst"
SKIP = {"index-append", "scroll"}


def _render(path):
    import jinja2
    return jinja2.Template(open(path).read()).render(distribution_version="3.9.0", index_name="big5", bulk_size=500, ingest_percentage=100)


def _mappings(d=None):
    d = d or os.environ.get("BIG5_DIR", "/data/big5")
    return json.loads(_render(os.path.join(d, "index.json")))["mappings"]


_MAP = os.path.join(os.environ.get("BIG5_DIR", "/data/big5"), "mapping.json")  # prep renders the template once (jinja2 is not importable under -I)
MAPPING = json.load(open(_MAP)) if os.path.exists(_MAP) else {}


def prep(a):
    src, out, n = a[0], a[1], int(a[2])
    import zstandard
    os.makedirs(out, exist_ok=True)
    batch, files, docs = [], [], 0
    with open(os.path.join(src, "big5-head.zst"), "rb") as fh:
        rd = zstandard.ZstdDecompressor().stream_reader(fh)
        buf = b""
        try:
            while docs < n:
                chunk = rd.read(1 << 22)
                if not chunk:
                    break
                buf += chunk
                lines = buf.split(b"\n")
                buf = lines.pop()
                for l in lines:
                    if docs >= n:
                        break
                    batch.append('{"index":{}}')
                    batch.append(l.decode())
                    docs += 1
                    if len(batch) >= 10000:
                        p = os.path.join(out, "bulk-%05d.ndjson" % len(files))
                        open(p, "w").write("\n".join(batch) + "\n")
                        files.append(p)
                        batch.clear()
        except zstandard.ZstdError:
            pass  # the range ends mid-frame
    if batch:
        p = os.path.join(out, "bulk-%05d.ndjson" % len(files))
        open(p, "w").write("\n".join(batch) + "\n")
        files.append(p)
    ops = json.loads("[" + _render(os.path.join(src, "ops.json")) + "]")
    Q = {}
    bodies = {}
    for o in ops:
        if o.get("operation-type") != "search" or o["name"] in SKIP or "body" not in o:
            continue
        Q[o["name"]] = [o["name"]] * int(os.environ.get("REPEAT", "20"))
        bodies[o["name"]] = o["body"]
    json.dump(_mappings(src), open(os.path.join(src, "mapping.json"), "w"))
    json.dump(Q, open(os.path.join(out, "queries.json"), "w"))
    json.dump(bodies, open(os.path.join(out, "bodies.json"), "w"))
    print(f"{docs} docs in {len(files)} bulk files, {len(Q)} operations")


_BODIES = {}


def body_for(name, q):
    if not _BODIES:
        _BODIES.update(json.load(open(os.path.join(os.environ.get("OUT_DIR", "/data/big5"), "bodies.json"))))
    b = dict(_BODIES[name])
    b.setdefault("size", 10)
    return b
