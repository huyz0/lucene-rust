"""Ranked text search (no aggregations) on a real corpus, stock vs plugin, on a node started by native-node.sh.

  text_bench.py prep SRC.jsonl OUT_DIR [N]   MS MARCO style jsonl ({"text": ...}) -> bulk files + queries.json
  text_bench.py load BASE OUT_DIR [SETTINGS_JSON]   index `txt`, force-merge to one segment
  text_bench.py time BASE LABEL OUT OUT_DIR   time every shape, many distinct queries each (PASSES timed passes)
  text_bench.py report GLOB_STOCK GLOB_OTHER...   best median per shape

Shapes are built in prep from the corpus's own term statistics (and, if QUERIES_TSV is set, from real
MS MARCO dev queries), so frequent / mid / rare terms, bigrams and multi-term queries are real ones.
The client is a keep-alive raw socket, so the HTTP floor is small and the same for both sides."""
import collections
import glob
import json
import os
import random
import re
import socket
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../opensearch-plugin/e2e"))
import benchhttp as bh  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")
SIZE = int(os.environ.get("SIZE", "10"))
PASSES = int(os.environ.get("PASSES", "5"))
WARMUP_S = float(os.environ.get("WARMUP_S", "3"))  # a freshly started JVM is still compiling after 30 requests
NQ = int(os.environ.get("NQ", "60"))


class Conn:
    """benchhttp's keep-alive client with this file's (parsed body, response bytes) return."""

    def __init__(self, base):
        self.c = bh.Client(base)

    def call(self, method, path, body=None):
        status, data = self.c.request(method, path, body)
        if status >= 300:
            raise RuntimeError(f"{method} {path}: {status} {data[:300]!r}")
        return json.loads(data), len(data)


def prep(src, out, n):
    os.makedirs(out, exist_ok=True)
    rnd = random.Random(7)
    df = collections.Counter()
    bi = collections.Counter()
    docs = []
    with open(src) as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            d = json.loads(line)
            docs.append(d["text"])
    files = []
    for f, start in enumerate(range(0, len(docs), 5000)):
        lines = []
        for i in range(start, min(len(docs), start + 5000)):
            lines.append('{"index":{"_id":"%d"}}' % i)
            lines.append(json.dumps({"text": docs[i], "n": i, "cat": "c%d" % (i % 20)}, separators=(",", ":")))
        p = os.path.join(out, "bulk-%05d.ndjson" % f)
        open(p, "w").write("\n".join(lines) + "\n")
        files.append(p)
    for t in docs:
        w = TOK.findall(t.lower())
        df.update(set(w))
        bi.update(set(zip(w, w[1:])))
    N = len(docs)
    ranked = [t for t, _ in df.most_common()]

    def band(lo, hi):  # terms with lo <= df/N < hi
        return [t for t in ranked if lo <= df[t] / N < hi and len(t) > 2]

    freq, mid, rare = band(0.02, 0.5), band(0.0005, 0.005), band(0.00001, 0.0001)
    low = [t for t in ranked if 20 <= df[t] <= 60 and len(t) > 3]
    Q = {}

    def pick(pool, k=1):
        return " ".join(rnd.sample(pool, k))

    Q["floor: match_none size 0"] = ["x"] * NQ
    Q["floor: rare term size 0"] = [pick(low) for _ in range(NQ)]
    Q["term frequent"] = [pick(freq) for _ in range(NQ)]
    Q["term mid"] = [pick(mid) for _ in range(NQ)]
    Q["term rare"] = [pick(low) for _ in range(NQ)]
    Q["match 2 mid (OR)"] = [pick(mid, 2) for _ in range(NQ)]
    Q["match frequent+mid (OR)"] = [pick(freq) + " " + pick(mid) for _ in range(NQ)]
    Q["match 4 mixed (OR)"] = [pick(freq) + " " + pick(mid, 2) + " " + pick(low) for _ in range(NQ)]
    Q["match 2 mid (AND)"] = [pick(mid, 2) for _ in range(NQ)]
    Q["match frequent+mid (AND)"] = [pick(freq) + " " + pick(mid) for _ in range(NQ)]
    # a phrase taken from a real document: 2, 3 words
    ph2, ph3, ph2f = [], [], []
    topbi = [b for b, c in bi.most_common(400) if all(len(x) > 1 for x in b)]
    while len(ph2) < NQ:
        w = TOK.findall(rnd.choice(docs).lower())
        if len(w) > 12:
            j = rnd.randrange(len(w) - 3)
            if df[w[j]] > 30 and df[w[j + 1]] > 30:
                ph2.append(" ".join(w[j:j + 2]))
                ph3.append(" ".join(w[j:j + 3]))
    Q["phrase 2 words (real)"] = ph2
    Q["phrase 3 words (real)"] = ph3
    Q["phrase 2 words (frequent bigram)"] = [" ".join(rnd.choice(topbi)) for _ in range(NQ)]
    Q["keyword term"] = ["c%d" % rnd.randrange(20) for _ in range(NQ)]
    Q["count (exact total, size 0)"] = [pick(freq) for _ in range(NQ)]
    Q["bool must+should+must_not"] = [json.dumps([pick(mid), pick(mid), pick(freq)]) for _ in range(NQ)]
    Q["bool must + filter cat"] = [pick(mid) for _ in range(NQ)]
    Q["bool must frequent + range n"] = [pick(freq) for _ in range(NQ)]
    qt = os.environ.get("QUERIES_TSV")
    if qt:
        real = []
        for line in open(qt):
            q = line.rstrip("\n").split("\t")[1]
            if len(TOK.findall(q.lower())) >= 2:
                real.append(q)
        rnd.shuffle(real)
        Q["real queries (OR)"] = real[:NQ * 2]
        Q["real queries (AND)"] = real[:NQ * 2]
        Q["real queries (75% should match)"] = real[:NQ * 2]
        Q["real queries + size 100"] = real[:NQ * 2]
        Q["real queries + _source false"] = real[:NQ * 2]
        Q["real queries + track_total_hits true"] = real[:NQ * 2]
    json.dump(Q, open(os.path.join(out, "queries.json"), "w"), indent=1)
    print(f"{N} docs, vocab {len(df)}, shapes {len(Q)}")


MAPPING = {"properties": {"text": {"type": "text"}, "n": {"type": "long"}, "cat": {"type": "keyword"}}}


def nstats(c):
    try:
        return c.call("GET", "/_plugins/lucene_rust/stats")[0]
    except RuntimeError:
        return None


def body_for(name, q):
    if name == "floor: match_none size 0":
        return {"size": 0, "query": {"match_none": {}}}
    if name == "floor: rare term size 0":
        return {"size": 0, "query": {"match": {"text": q}}}
    if name == "keyword term":
        return {"size": SIZE, "query": {"term": {"cat": q}}}
    if name.startswith("count"):
        return {"size": 0, "track_total_hits": True, "query": {"match": {"text": q}}}
    if name.startswith("bool must+should"):
        a, b, c = json.loads(q)
        return {"size": SIZE, "query": {"bool": {"must": {"match": {"text": a}}, "should": {"match": {"text": b}}, "must_not": {"match": {"text": c}}}}}
    base = {"size": SIZE}
    if name.endswith("(AND)"):
        base["query"] = {"match": {"text": {"query": q, "operator": "and"}}}
    elif "75%" in name:
        base["query"] = {"match": {"text": {"query": q, "minimum_should_match": "75%"}}}
    elif name.startswith("phrase"):
        base["query"] = {"match_phrase": {"text": q}}
    elif name == "bool must + filter cat":
        base["query"] = {"bool": {"must": {"match": {"text": q}}, "filter": {"term": {"cat": "c3"}}}}
    elif name == "bool must frequent + range n":
        base["query"] = {"bool": {"must": {"match": {"text": q}}, "filter": {"range": {"n": {"gte": 100000, "lt": 400000}}}}}
    else:
        base["query"] = {"match": {"text": q}}
    if name.endswith("size 100"):
        base["size"] = 100
    if name.endswith("_source false"):
        base["_source"] = False
    if name.endswith("track_total_hits true"):
        base["track_total_hits"] = True
    return base


def load(base, out, settings):
    import concurrent.futures
    c = Conn(base)
    try:
        c.call("DELETE", "/txt")
    except RuntimeError:
        pass
    st = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    st.update(json.loads(settings))
    c.call("PUT", "/txt", {"settings": st, "mappings": MAPPING})
    files = sorted(glob.glob(os.path.join(out, "bulk-*.ndjson")))
    t0 = time.perf_counter()

    def send(p):
        r, _ = Conn(base).call("POST", "/txt/_bulk", open(p, "rb").read())
        if r["errors"]:
            raise RuntimeError("bulk errors")

    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        list(ex.map(send, files))
    t1 = time.perf_counter() - t0
    c.call("POST", "/txt/_refresh")
    c.call("POST", "/txt/_forcemerge?max_num_segments=1")
    c.call("POST", "/txt/_refresh")
    idx = c.call("GET", "/_cat/indices/txt?format=json&bytes=b")[0][0]
    print(f"{c.call('GET', '/txt/_count')[0]['count']} docs in {t1:.0f}s, store {int(idx['store.size']) / 2**20:.0f} MiB")


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
        bh.warm(lambda: c.call("POST", "/txt/_search?request_cache=false", bodies[next(it) % len(bodies)]),
                len(bodies), WARMUP_S)
        wall, took, size = [], [], 0
        s0 = nstats(c)
        for _ in range(PASSES):
            for b in bodies:
                t = time.perf_counter()
                r, n = c.call("POST", "/txt/_search?request_cache=false", b)
                wall.append((time.perf_counter() - t) * 1000)
                took.append(r["took"])
                size += n
        wall.sort()
        s1 = nstats(c)
        nat = {}
        if s0 and s1:
            n = len(wall)
            for k in ("query_phase_nanos", "fetch_nanos"):
                a, b = s0[k], s1[k]
                nat[k.split("_")[0] + "_native_ms"] = round((b["accelerated"] - a["accelerated"]) / n / 1e6, 3)
                nat[k.split("_")[0] + "_fallback_ms"] = round((b["fallback"] - a["fallback"]) / n / 1e6, 3)
        res[name] = {**nat, "median_ms": round(statistics.median(wall), 3), "mean_ms": round(statistics.mean(wall), 3),
                     "p90_ms": round(wall[int(len(wall) * 0.9) - 1], 3), "p99_ms": round(bh.pct(wall, 0.99), 3), "n": len(wall), "took_mean": round(statistics.mean(took), 2),
                     "resp_bytes": size // len(wall)}
    json.dump(res, open(outf, "w"), indent=1)


def report(pats):
    import re as _re
    groups = {}
    for p in pats:
        for g in sorted(glob.glob(p)):
            groups.setdefault(_re.sub(r"-r\d+\.json$", "", g), []).append(g)
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
    if a[1] == "prep":
        prep(a[2], a[3], int(a[4]) if len(a) > 4 else 1000000)
    elif a[1] == "load":
        load(a[2], a[3], a[4] if len(a) > 4 else "{}")
    elif a[1] == "time":
        time_all(a[2], a[3], a[4], a[5])
    elif a[1] == "loop":  # loop BASE OUT_DIR SHAPE SECONDS PID: closed loop of one shape, prints qps and cpu ms/req
        Q = json.load(open(os.path.join(a[3], "queries.json")))
        bodies = [json.dumps(body_for(a[4], q)).encode() for q in Q[a[4]]]
        c = Conn(a[2])
        for b in bodies:
            c.call("POST", "/txt/_search?request_cache=false", b)

        def cpu():
            f = open(f"/proc/{a[6]}/stat").read().rsplit(")", 1)[1].split()
            return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")

        c0, t0, n = cpu(), time.perf_counter(), 0
        while time.perf_counter() - t0 < float(a[5]):
            c.call("POST", "/txt/_search?request_cache=false", bodies[n % len(bodies)])
            n += 1
        dt = time.perf_counter() - t0
        print(json.dumps({"shape": a[4], "qps": round(n / dt), "cpu_ms_per_req": round((cpu() - c0) / n * 1000, 3)}))
    elif a[1] == "dump":  # dump BASE OUT_DIR SHAPE FILE: top-10 ids and scores of every query, to compare engines
        Q = json.load(open(os.path.join(a[3], "queries.json")))
        c = Conn(a[2])
        with open(a[5], "w") as fh:
            for q in Q[a[4]]:
                r, _ = c.call("POST", "/txt/_search?request_cache=false", body_for(a[4], q))
                fh.write(json.dumps([[h["_id"], h["_score"]] for h in r["hits"]["hits"]]) + "\n")
    elif a[1] == "report":
        report(a[2:])
    else:
        sys.exit(__doc__)
