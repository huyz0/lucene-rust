"""prof_table.py [DIR]: per shape, node CPU per request and where it goes, from prof.sh's output.

Stock and plugin side by side: ms of CPU per request split into the query phase (Lucene or native), fetch
(`_source`, stored fields), request plumbing (REST, netty, coordination, shard context, response, thread
pool, the plugin's Java glue) and JIT/GC/other. The plugin's native engine bucket holds query and fetch both."""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import buckets  # noqa: E402

GROUP = {
    "query phase (Lucene/Java)": "query", "native engine (Rust)": "query",
    "fetch: _source filter/parse": "fetch", "fetch: stored fields / docs": "fetch",
    "gc/jit/vm": "jit/gc", "other": "other",
}


def split(path):
    tot = 0
    g = collections.Counter()
    for line in open(path, errors="replace"):
        stack, _, n = line.rstrip().rpartition(" ")
        if not stack:
            continue
        n = int(n)
        tot += n
        g[GROUP.get(buckets.bucket(stack), "plumbing")] += n
    return {k: v / tot for k, v in g.items()} if tot else {}


def main(d):
    rows = {}
    for kind in ("stock", "plugin"):
        for line in open(os.path.join(d, f"{kind}-loop.jsonl")):
            r = json.loads(line)
            rows.setdefault(r["shape"], {})[kind] = r
        for i, shape in enumerate(rows, 1):
            pass
    names = {}
    for kind in ("stock", "plugin"):
        i = 1
        while os.path.exists(os.path.join(d, f"{kind}-{i}.name")):
            names[(kind, open(os.path.join(d, f"{kind}-{i}.name")).read().strip())] = os.path.join(d, f"{kind}-{i}.collapsed")
            i += 1
    print("| shape | engine | cpu ms/req | query | fetch | plumbing | jit/gc/other |")
    print("|---|---|---|---|---|---|---|")
    for shape, r in rows.items():
        for kind in ("stock", "plugin"):
            if kind not in r or (kind, shape) not in names:
                continue
            s = split(names[(kind, shape)])
            c = r[kind]["cpu_ms_per_req"]
            f = lambda k: f"{c * s.get(k, 0):.2f}"
            print(f"| {shape} | {kind} | {c:.2f} | {f('query')} | {f('fetch')} | {f('plumbing')} | {c * (s.get('jit/gc', 0) + s.get('other', 0)):.2f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/prof")
