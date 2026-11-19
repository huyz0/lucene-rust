"""Ranked text search on long documents (English Wikipedia articles, `title` + `text`), multi-shard, serial and concurrent.

  wiki_bench.py prep PARQUET OUT_DIR [N]        Wikipedia parquet (HF wikimedia/wikipedia) -> bulk files + queries.json
  wiki_bench.py load BASE OUT_DIR [SETTINGS]    index `wiki` (SHARDS shards, default 4), force-merge to 1 segment per shard
  wiki_bench.py time BASE LABEL OUT OUT_DIR     serial latency per shape (PASSES timed passes over the distinct queries)
  wiki_bench.py conc BASE OUT_DIR SHAPE SECS PID [PROCS=1] [CONNS=8] [WARM=3]
                                                closed-loop keep-alive clients: qps, p50/p99 ms, node CPU ms per request
  wiki_bench.py report GLOB_STOCK GLOB_OTHER... best mean per shape (serial) or best qps per shape (conc files)

Queries come from the corpus itself (no external query log exists for Wikipedia): term bands by document
frequency, real bigrams/trigrams from documents, 6-word fragments of documents (a stand-in for natural-language
queries), and article titles (entity lookups). `cat` (keyword, id % 20) and `n` (long, doc number) are synthetic
fields for the filter shapes. The client is benchhttp's keep-alive raw socket."""
import asyncio
import collections
import glob
import json
import multiprocessing as mp
import os
import random
import re
import statistics
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../opensearch-plugin/e2e"))
import benchhttp as bh  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")
SIZE = int(os.environ.get("SIZE", "10"))
PASSES = int(os.environ.get("PASSES", "3"))
WARMUP_S = float(os.environ.get("WARMUP_S", "3"))
NQ = int(os.environ.get("NQ", "60"))
SHARDS = int(os.environ.get("SHARDS", "4"))
SAMPLE_EVERY = int(os.environ.get("SAMPLE_EVERY", "5"))  # term statistics come from every Nth document
INDEX = "wiki"


class Conn:
    def __init__(self, base):
        self.c = bh.Client(base)

    def call(self, method, path, body=None):
        status, data = self.c.request(method, path, body)
        if status >= 300:
            raise RuntimeError(f"{method} {path}: {status} {data[:300]!r}")
        return json.loads(data), len(data)


def prep(src, out, n):
    import pyarrow.parquet as pq
    os.makedirs(out, exist_ok=True)
    rnd = random.Random(11)
    f = pq.ParquetFile(src)
    titles, sample = [], []
    N = 0
    batch = []
    files = []

    def flush():
        p = os.path.join(out, "bulk-%05d.ndjson" % len(files))
        open(p, "w").write("\n".join(batch) + "\n")
        files.append(p)
        batch.clear()

    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=["title", "text"]).to_pydict()
        for title, text in zip(t["title"], t["text"]):
            if N >= n:
                break
            if len(text) < 200:  # stubs
                continue
            batch.append('{"index":{"_id":"%d"}}' % N)
            batch.append(json.dumps({"title": title, "text": text, "n": N, "cat": "c%d" % (N % 20)}, separators=(",", ":")))
            titles.append(title)
            if N % SAMPLE_EVERY == 0:
                sample.append(text)
            N += 1
            if len(batch) >= 1000:  # 500 docs per bulk
                flush()
        if N >= n:
            break
    if batch:
        flush()
    df = collections.Counter()
    bi = collections.Counter()
    for i, t in enumerate(sample):
        w = TOK.findall(t.lower())
        df.update(set(w))
        if i % 3 == 0:
            bi.update(set(zip(w, w[1:])))
    S = len(sample)
    ranked = [t for t, _ in df.most_common()]

    def band(lo, hi):
        return [t for t in ranked if lo <= df[t] / S < hi and len(t) > 2 and t.isalpha()]

    freq, mid, rare = band(0.02, 0.4), band(0.001, 0.01), band(0.00005, 0.0003)
    print(f"{N} docs, sample {S}, vocab {len(df)}, bands freq {len(freq)} mid {len(mid)} rare {len(rare)}")

    def pick(pool, k=1):
        return " ".join(rnd.sample(pool, k))

    docs_tok = [TOK.findall(t.lower()) for t in rnd.sample(sample, 3000)]

    def fragment(k, need_df=0):
        while True:
            w = rnd.choice(docs_tok)
            if len(w) > 200:
                j = rnd.randrange(50, len(w) - k - 1)
                frag = w[j:j + k]
                if all(df[x] > need_df for x in frag):
                    return frag

    Q = {}
    Q["floor: match_none size 0"] = ["x"] * NQ
    Q["term frequent"] = [pick(freq) for _ in range(NQ)]
    Q["term mid"] = [pick(mid) for _ in range(NQ)]
    Q["term rare"] = [pick(rare) for _ in range(NQ)]
    Q["match 2 mid (OR)"] = [pick(mid, 2) for _ in range(NQ)]
    Q["match frequent+mid (OR)"] = [pick(freq) + " " + pick(mid) for _ in range(NQ)]
    Q["match 4 mixed (OR)"] = [pick(freq) + " " + pick(mid, 2) + " " + pick(rare) for _ in range(NQ)]
    Q["match 2 mid (AND)"] = [pick(mid, 2) for _ in range(NQ)]
    Q["match frequent+mid (AND)"] = [pick(freq) + " " + pick(mid) for _ in range(NQ)]
    Q["phrase 2 words (real)"] = [" ".join(fragment(2, 5)) for _ in range(NQ)]
    Q["phrase 3 words (real)"] = [" ".join(fragment(3, 5)) for _ in range(NQ)]
    topbi = [b for b, c in bi.most_common(500) if all(len(x) > 1 for x in b)]
    Q["phrase 2 words (frequent bigram)"] = [" ".join(rnd.choice(topbi)) for _ in range(NQ)]
    Q["bool must+should+must_not"] = [json.dumps([pick(mid), pick(mid), pick(freq)]) for _ in range(NQ)]
    Q["bool must + filter cat"] = [pick(mid) for _ in range(NQ)]
    Q["bool must frequent + range n"] = [pick(freq) for _ in range(NQ)]
    frag = [" ".join(fragment(6, 5)) for _ in range(NQ * 2)]
    Q["fragments 6 words (OR)"] = frag
    Q["fragments 6 words (AND)"] = frag
    Q["fragments 6 words (75% should match)"] = frag
    Q["fragments 6 words + size 100"] = frag
    Q["fragments 6 words + _source false"] = frag
    Q["fragments 6 words + track_total_hits true"] = frag
    tt = [t for t in titles if 1 <= len(TOK.findall(t.lower())) <= 4]
    tq = rnd.sample(tt, NQ * 2)
    Q["titles on text (OR)"] = tq
    Q["titles multi_match title^3,text"] = tq
    Q["titles on title (AND)"] = tq
    Q["count (exact total, size 0)"] = [pick(freq) for _ in range(NQ)]
    json.dump(Q, open(os.path.join(out, "queries.json"), "w"), indent=1)
    print(f"shapes {len(Q)}")


MAPPING = {"properties": {"title": {"type": "text"}, "text": {"type": "text"}, "n": {"type": "long"}, "cat": {"type": "keyword"}}}


def body_for(name, q):
    if name == "floor: match_none size 0":
        return {"size": 0, "query": {"match_none": {}}}
    if name.startswith("count"):
        return {"size": 0, "track_total_hits": True, "query": {"match": {"text": q}}}
    if name.startswith("bool must+should"):
        a, b, c = json.loads(q)
        return {"size": SIZE, "query": {"bool": {"must": {"match": {"text": a}}, "should": {"match": {"text": b}}, "must_not": {"match": {"text": c}}}}}
    base = {"size": SIZE}
    if name.startswith("titles multi_match"):
        base["query"] = {"multi_match": {"query": q, "fields": ["title^3", "text"]}}
    elif name.startswith("titles on title"):
        base["query"] = {"match": {"title": {"query": q, "operator": "and"}}}
    elif name.endswith("(AND)"):
        base["query"] = {"match": {"text": {"query": q, "operator": "and"}}}
    elif "75%" in name:
        base["query"] = {"match": {"text": {"query": q, "minimum_should_match": "75%"}}}
    elif name.startswith("phrase"):
        base["query"] = {"match_phrase": {"text": q}}
    elif name == "bool must + filter cat":
        base["query"] = {"bool": {"must": {"match": {"text": q}}, "filter": {"term": {"cat": "c3"}}}}
    elif name == "bool must frequent + range n":
        base["query"] = {"bool": {"must": {"match": {"text": q}}, "filter": {"range": {"n": {"gte": 30000, "lt": 100000}}}}}
    else:
        base["query"] = {"match": {"text": q}}
    if name.endswith("size 100"):
        base["size"] = 100
    if name.endswith("_source false"):
        base["_source"] = False
    if name.endswith("track_total_hits true"):
        base["track_total_hits"] = True
    return base


def nstats(c):
    try:
        return c.call("GET", "/_plugins/accelerated/stats")[0]
    except RuntimeError:
        return None


def _node_cpu():
    import subprocess
    try:
        return _cpu(subprocess.check_output(["pgrep", "-f", "org.opensearch.bootstrap.OpenSearch"]).split()[0].decode())
    except Exception:
        return 0.0


def load(base, out, settings):
    import concurrent.futures
    c = Conn(base)
    try:
        c.call("DELETE", "/" + INDEX)
    except RuntimeError:
        pass
    st = {"number_of_shards": SHARDS, "number_of_replicas": 0, "refresh_interval": -1}
    st.update(json.loads(settings))
    c.call("PUT", "/" + INDEX, {"settings": st, "mappings": MAPPING})
    files = sorted(glob.glob(os.path.join(out, "bulk-*.ndjson")))
    t0 = time.perf_counter()
    cpu0 = _node_cpu()

    def send(p):
        r, _ = Conn(base).call("POST", f"/{INDEX}/_bulk", open(p, "rb").read())
        if r["errors"]:
            raise RuntimeError("bulk errors")

    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        list(ex.map(send, files))
    t1 = time.perf_counter() - t0
    cpu1 = _node_cpu()
    c.call("POST", f"/{INDEX}/_refresh")
    t2 = time.perf_counter()
    if os.environ.get("NOMERGE"):  # leave the segments the merge policy made while loading (refresh_interval in SETTINGS)
        time.sleep(40)
        c.call("PUT", f"/{INDEX}/_settings", {"index": {"refresh_interval": -1}})
        c.call("POST", f"/{INDEX}/_refresh")
        segs = c.call("GET", f"/_cat/segments/{INDEX}?format=json")[0]
        print(f"{len(segs)} segments across {SHARDS} shards")
    else:
        c.call("POST", f"/{INDEX}/_forcemerge?max_num_segments=1")
        c.call("POST", f"/{INDEX}/_refresh")
    idx = c.call("GET", f"/_cat/indices/{INDEX}?format=json&bytes=b")[0][0]
    ndocs = c.call('GET', f'/{INDEX}/_count')[0]['count']
    cpu2 = _node_cpu()
    print(f"{ndocs} docs in {t1:.0f}s = {ndocs / t1:.0f} docs/s (+merge {time.perf_counter() - t2:.0f}s), "
          f"{SHARDS} shards, store {int(idx['store.size']) / 2**20:.0f} MiB, node CPU {cpu1 - cpu0:.1f}s bulk + {cpu2 - cpu1:.1f}s merge")
    print(json.dumps({"load": {"docs": ndocs, "bulk_s": round(t1, 1), "docs_per_s": round(ndocs / t1), "merge_s": round(time.perf_counter() - t2, 1),
                               "store_mib": round(int(idx['store.size']) / 2**20), "cpu_bulk_s": round(cpu1 - cpu0, 1), "cpu_merge_s": round(cpu2 - cpu1, 1)}}))


def time_all(base, label, outf, out):
    Q = json.load(open(os.path.join(out, "queries.json")))
    want = [w for w in os.environ.get("SHAPES", "").split(",") if w]
    c = Conn(base)
    res = {}
    for name, qs in Q.items():
        if want and not any(w in name for w in want):
            continue
        bodies = [json.dumps(body_for(name, q)).encode() for q in qs]
        it = iter(range(10**9))
        path = f"/{INDEX}/_search?request_cache=false"
        bh.warm(lambda: c.call("POST", path, bodies[next(it) % len(bodies)]), len(bodies), WARMUP_S)
        wall, took, size = [], [], 0
        s0 = nstats(c)
        for _ in range(PASSES):
            for b in bodies:
                t = time.perf_counter()
                r, n = c.call("POST", path, b)
                wall.append((time.perf_counter() - t) * 1000)
                took.append(r["took"])
                size += n
        wall.sort()
        s1 = nstats(c)
        nat = {}
        if s0 and s1:
            n = len(wall)
            for k in ("query_phase_nanos", "fetch_nanos"):
                nat[k.split("_")[0] + "_native_ms"] = round((s1[k]["accelerated"] - s0[k]["accelerated"]) / n / 1e6, 3)
                nat[k.split("_")[0] + "_fallback_ms"] = round((s1[k]["fallback"] - s0[k]["fallback"]) / n / 1e6, 3)
        res[name] = {**nat, "median_ms": round(statistics.median(wall), 3), "mean_ms": round(statistics.mean(wall), 3),
                     "p90_ms": round(wall[int(len(wall) * 0.9) - 1], 3), "p99_ms": round(bh.pct(wall, 0.99), 3), "n": len(wall),
                     "took_mean": round(statistics.mean(took), 2), "resp_bytes": size // len(wall)}
    json.dump(res, open(outf, "w"), indent=1)


async def _worker(host, port, reqs, start, stop_at, warm_at, lat, idx):
    r, w = await asyncio.open_connection(host, port)
    i = idx
    while time.perf_counter() < stop_at:
        body = reqs[i % len(reqs)]
        i += 7  # co-prime stride so connections do not run in lockstep
        t = time.perf_counter()
        w.write(body)
        await w.drain()
        head = await r.readuntil(b"\r\n\r\n")
        n = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                n = int(line.split(b":")[1])
        await r.readexactly(n)
        if t >= warm_at:
            lat.append(time.perf_counter() - t)
    w.close()


def _proc(args):
    host, port, reqs, start, stop_at, warm_at, conns, q = args
    lat = []

    async def main():
        await asyncio.gather(*[_worker(host, port, reqs, start, stop_at, warm_at, lat, k) for k in range(conns)])

    asyncio.run(main())
    q.put(lat)


def _cpu(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def conc(base, out, shape, secs, pid, procs, conns, warm):
    u = urlparse(base)
    Q = json.load(open(os.path.join(out, "queries.json")))
    reqs = []
    for q in Q[shape]:
        b = json.dumps(body_for(shape, q)).encode()
        reqs.append((f"POST /{INDEX}/_search?request_cache=false HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(b)}\r\n\r\n").encode() + b)
    qu = mp.Queue()
    t0 = time.perf_counter() + 1
    warm_at, stop_at = t0 + warm, t0 + warm + secs
    ps = [mp.Process(target=_proc, args=((u.hostname, u.port, reqs[k::procs] or reqs, t0, stop_at, warm_at, conns, qu),))
          for k in range(procs)]
    for p in ps:
        p.start()
    time.sleep(max(0, warm_at - time.perf_counter()))
    c0, tt = _cpu(pid), time.perf_counter()
    lats = [qu.get() for _ in ps]
    for p in ps:
        p.join()
    c1, dt = _cpu(pid), time.perf_counter() - tt
    lat = sorted(x * 1000 for l in lats for x in l)
    n = len(lat)
    print(json.dumps({"shape": shape, "clients": procs * conns, "qps": round(n / dt, 1), "p50_ms": round(statistics.median(lat), 2),
                      "p99_ms": round(bh.pct(lat, 0.99), 2), "cpu_ms_per_req": round((c1 - c0) / n * 1000, 3), "n": n}))


def report(pats):
    groups = {}
    for p in pats:
        for g in sorted(glob.glob(p)):
            groups.setdefault(re.sub(r"-r\d+\.json$", "", g), []).append(g)
    names = list(groups)
    runs = {n: [json.load(open(f)) for f in groups[n]] for n in names}
    best = {n: {k: min(r[k]["mean_ms"] for r in runs[n] if k in r) for k in runs[n][0]} for n in names}
    print(f"{'shape':40s} " + " ".join(f"{os.path.basename(n)[:12]:>12s}" for n in names) + "   (best mean ms; ratio = stock/other)")
    for k in best[names[0]]:
        row = f"{k:40s} " + " ".join(f"{best[n].get(k, 0):12.2f}" for n in names)
        row += "  " + " ".join(f"{best[names[0]][k] / best[n][k]:.2f}x" for n in names[1:] if best[n].get(k))
        print(row)


if __name__ == "__main__":
    a = sys.argv
    if os.environ.get("BENCH_EXT"):  # a shape set: INDEX, MAPPING, body_for(name, q), prep(argv) (see confluence_shapes.py)
        import importlib.util
        sp = importlib.util.spec_from_file_location("ext", os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ["BENCH_EXT"] + ".py"))
        ext = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(ext)
        INDEX, MAPPING, body_for = ext.INDEX, ext.MAPPING, ext.body_for
        if a[1] == "prep":
            ext.prep(a[2:])
            sys.exit(0)
    if a[1] == "prep":
        prep(a[2], a[3], int(a[4]) if len(a) > 4 else 10**9)
    elif a[1] == "load":
        load(a[2], a[3], a[4] if len(a) > 4 else "{}")
    elif a[1] == "time":
        time_all(a[2], a[3], a[4], a[5])
    elif a[1] == "conc":
        conc(a[2], a[3], a[4], float(a[5]), a[6], int(a[7]) if len(a) > 7 else 1, int(a[8]) if len(a) > 8 else 8,
             float(a[9]) if len(a) > 9 else 3)
    elif a[1] == "dump":  # dump BASE OUT_DIR SHAPE FILE: top-10 ids and scores of every query, to compare engines
        Q = json.load(open(os.path.join(a[3], "queries.json")))
        c = Conn(a[2])
        with open(a[5], "w") as fh:
            for q in Q[a[4]]:
                r, _ = c.call("POST", f"/{INDEX}/_search?request_cache=false", body_for(a[4], q))
                fh.write(json.dumps([r["hits"]["total"], [[h["_id"], h["_score"]] for h in r["hits"]["hits"]]]) + "\n")
    elif a[1] == "report":
        report(a[2:])
    else:
        sys.exit(__doc__)
