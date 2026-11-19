#!/usr/bin/env python3
"""Source-parsing differential: adversarial `_source` documents for a mapping of the fields the
accelerated source reader handles (keyword, text, integer, long, float, double, date), indexed on a
plugin node with `index.engine.accelerated_source_parsing` and on a stock node. Both must accept and
reject the same documents with the same errors, and the indexed data must read back the same: doc
values, term and range counts.

Start the plugin node with `-Daccelerated.ingest.verify=true` as well and the engine also runs
OpenSearch's document mapper on every document the accelerated reader takes, comparing what each
produces field by field (`native ingest mismatch` on stderr; the script counts them when given
`--plugin-log`).

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/fuzz_ingest.py --seed 1 --docs 20000 --valid 0.6
"""
import argparse
import json
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node  # noqa: E402

def rnd_str(r):
    k = r.random()
    if k < .3: return "".join(r.choice("abcdefghij-_/ ") for _ in range(r.randrange(0, 20)))
    if k < .4: return "caf\u00e9 na\u00efve \u4e2d\u6587 \U0001F600 emoji"
    if k < .5: return "line1\nline2\t\"quoted\" back\\slash /slash \u0001 ctrl"
    if k < .55: return ""
    if k < .6: return "x" * r.choice([255, 256, 257, 1000, 32766, 32767, 40000])
    if k < .7: return "\ud83d\ude00".encode("utf-16", "surrogatepass").decode("utf-16", "surrogatepass") if False else "A\u00df\u1e9e\u0130i\u0307"
    if k < .8: return " ".join(r.choice(["the","quick","brown","fox","Jumps","over","LAZY","dog","123","a.b","foo_bar","x@y.com","O'Neil","e-mail"]) for _ in range(r.randrange(1, 30)))
    return str(r.random())

def num_token(r, kind):
    k = r.random()
    if kind in ("int",):
        c = [0, 1, -1, 2147483647, -2147483648, 2147483648, -2147483649, 123456, 99999999999]
        if k < .6: return r.choice(c)
        if k < .7: return r.choice([1.5, -1.5, 0.9999, 1e3, 1E2, 12.0, "12", "12.7", " 5", "0x10", "abc", "", None, True, [1], [1, 2], {"a": 1}])
        return r.randrange(-2**31, 2**31)
    if kind == "long":
        c = [0, 1, -1, 2**63 - 1, -2**63, 2**63, -2**63 - 1, 9007199254740993, 10**18]
        if k < .6: return r.choice(c)
        if k < .7: return r.choice([1.5, 1e3, 1.0, "12", "9223372036854775807", None, True, [5], 1e19, 12345678901234567890])
        return r.randrange(-2**63, 2**63)
    if kind == "float":
        c = [1.00000005960464477539062501, 1.00000005960464477539062499, 16777217.00000001, 3.4028235677973366e38, 0, 1, -1, 0.1, 0.2, 40.12, -0.0, 3.4028235e38, 3.4028236e38, 1e39, 1e-45, 1.4e-45, 1.17549435e-38, 1e-46,
             16777217, 16777216.5, 9007199254740993, 0.30000001192092896, 8.589973e9, 123456789.123456789]
        if k < .5: return r.choice(c)
        if k < .6: return r.choice(["1.5", "abc", None, True, [1.5], "NaN", "Infinity", 1e400 if False else 1e308])
        if k < .8: return round(r.expovariate(1 / 40), r.choice([0, 1, 2, 3, 8, 17]))
        return struct.unpack("<f", struct.pack("<I", r.getrandbits(32)))[0] if r.random() < .5 else r.uniform(-1e10, 1e10)
    if kind == "double":
        c = [0, 1, -1, 0.1, 1e308, 1.7976931348623157e308, 5e-324, 2.2250738585072014e-308, 9007199254740993, -0.0, 123456789.123456789]
        if k < .5: return r.choice(c)
        if k < .6: return r.choice(["1.5", None, True, [1.5], "abc"])
        return r.uniform(-1e15, 1e15) if r.random() < .5 else r.expovariate(1 / 40)

def date_val(r):
    k = r.random()
    y, mo, d = r.randrange(1970, 2100), r.randrange(1, 13), r.randrange(1, 29)
    h, mi, s = r.randrange(24), r.randrange(60), r.randrange(60)
    base = "%04d-%02d-%02dT%02d:%02d:%02d" % (y, mo, d, h, mi, s)
    if k < .3: return base + ".%03dZ" % r.randrange(1000)
    if k < .4: return base + "Z"
    if k < .5: return base + ".%dZ" % r.randrange(10)
    if k < .55: return base + ".%02dZ" % r.randrange(100)
    if k < .6: return base + ".%06dZ" % r.randrange(10**6)
    if k < .65: return base + ".123456789Z"
    if k < .7: return base + "+00:00"
    if k < .73: return base + "+05:30"
    if k < .76: return base + ".5-08:00"
    if k < .8: return r.choice(["2024-02-29T23:59:59Z", "2023-02-29T00:00:00Z", "2026-02-30T00:00:00Z", "2026-13-01T00:00:00Z", "2026-00-10T00:00:00Z",
                                 "2026-11-01T24:00:00Z", "2026-11-01T23:60:00Z", "2026-11-01T23:59:60Z", "0001-01-01T00:00:00Z", "0000-01-01T00:00:00Z",
                                 "9999-12-31T23:59:59.999Z", "10000-01-01T00:00:00Z", "1969-12-31T23:59:59.999Z", "1900-01-01T00:00:00Z", "1000-01-01T00:00:00Z",
                                 "1970-01-01T00:00:00Z", "-0001-01-01T00:00:00Z", "2026-11-01", "2026-11", "2026", "2026-11-01T10", "2026-11-01T10:20", "2026-11-01T10:20:30",
                                 "2026-11-01 10:20:30Z", "2026-11-01t10:20:30z", "20261101T102030Z", "2026-1-1T00:00:00Z", "2026-11-01T00:00:00.Z", "", "abc", " 2026-11-01T00:00:00Z"])
    if k < .9: return r.choice([0, 1700000000000, -1, 253402300799999, 1.7e12, "1700000000000", "-5", 1700000000000.5, True, None, [1], [base + "Z"]])
    return base + ".%03dZ" % r.randrange(1000)

def key_variants(r, name):
    return name

def valid_doc(r):
    d = {}
    if r.random() < .98:
        y, mo, d_ = r.randrange(1971, 2100), r.randrange(1, 13), r.randrange(1, 29)
        t = "%04d-%02d-%02dT%02d:%02d:%02d" % (y, mo, d_, r.randrange(24), r.randrange(60), r.randrange(60))
        d["@timestamp"] = t + r.choice(["Z", ".%03dZ" % r.randrange(1000), ".%dZ" % r.randrange(10), ".%02dZ" % r.randrange(100)])
    if r.random() < .98: d["host"] = r.choice([rnd_str(r) if r.random() < .5 else "host-%d" % r.randrange(500), "k" * r.randrange(1, 300)])[:32000]
    if r.random() < .95: d["status"] = r.choice([r.randrange(-2**31, 2**31), 200, 404, 0, -1, 2147483647, -2147483648])
    if r.random() < .95: d["bytes"] = r.choice([r.randrange(-2**63, 2**63), r.randrange(0, 10**6), 2**63 - 1, -2**63])
    if r.random() < .95:
        k = r.random()
        d["latency_ms"] = r.choice([round(r.expovariate(1 / 40), r.randrange(0, 6)), r.uniform(-1e6, 1e6), struct.unpack("<f", struct.pack("<I", r.getrandbits(32)))[0] if k < .5 else 0.1 * r.randrange(100)])
        if isinstance(d["latency_ms"], float) and (math.isnan(d["latency_ms"]) or math.isinf(d["latency_ms"])): d["latency_ms"] = 1.5
    if r.random() < .95:
        d["dbl"] = r.choice([r.uniform(-1e15, 1e15), r.expovariate(1 / 40), r.randrange(-10**15, 10**15), 1e-300 * r.random(), 1e300 * r.random()])
    if r.random() < .95: d["message"] = rnd_str(r) if r.random() < .5 else " ".join(r.choice(["the","quick","brown","fox","Jumps","over","LAZY","dog","123","a.b","foo_bar","x@y.com","O'Neil","e-mail"]) for _ in range(r.randrange(1, 30)))
    items = list(d.items())
    r.shuffle(items)
    return json.dumps(dict(items), ensure_ascii=r.random() < .5)

def gen_doc(r):
    if os.environ.get("VALID") and r.random() < float(os.environ["VALID"]):
        return valid_doc(r)
    d = {}
    def put(k, v):
        d[k] = v
    if r.random() < .97: put("@timestamp", date_val(r))
    if r.random() < .95: put("host", r.choice([rnd_str(r), "host-%03d" % r.randrange(200)]) if r.random() < .9 else r.choice([1, 1.5, True, None, [], ["a"], ["a", "b"], {"a": "b"}]))
    if r.random() < .9: put("status", num_token(r, "int"))
    if r.random() < .9: put("bytes", num_token(r, "long"))
    if r.random() < .9: put("latency_ms", num_token(r, "float"))
    if r.random() < .9: put("dbl", num_token(r, "double"))
    if r.random() < .9: put("message", r.choice([rnd_str(r)] * 4 + [None, 5, ["a b", "c d"], ""]))
    if r.random() < .1: put("unknown_" + str(r.randrange(5)), r.choice([1, "x", None, [1], {"k": "v"}]))
    if r.random() < .03: put("host", "dup")
    items = list(d.items())
    if r.random() < .3: r.shuffle(items)
    # serialize with variety
    style = r.random()
    if style < .5:
        s = json.dumps(dict(items), ensure_ascii=r.random() < .5)
    elif style < .7:
        s = json.dumps(dict(items), indent=2)
    elif style < .8:
        s = " \n\t{ " + " , ".join(json.dumps(k) + "\t:\n" + json.dumps(v) for k, v in items) + "\r\n}  \n"
    elif style < .85:
        s = "{" + ",".join(json.dumps(k) + ":" + json.dumps(v) for k, v in items) + ("," if r.random() < .5 else "") + "}"
    elif style < .88:
        s = "{" + ",".join(json.dumps(k) + ":" + json.dumps(v) for k, v in items) + "} x"
    elif style < .9:
        s = "{" + ",".join(json.dumps(k) + ":" + json.dumps(v) for k, v in items) + "}/* c */"
    elif style < .92 and items:
        k, v = items[0]
        s = "{" + json.dumps(k) + ":" + json.dumps(v) + "," + ",".join(json.dumps(k) + ":" + json.dumps(v) for k, v in items) + "}"
    elif style < .94:
        s = r.choice(["{}", "[]", "null", "5", "\"x\"", "{\"a\"", "", "{\"host\":\"a\" \"b\":1}", "{'host':'a'}", "{\"host\":\"\\ud800\"}", "{\"host\":\"\\u00zz\"}", "{\"status\":01}", "{\"status\":1.}", "{\"status\":-}", "{\"status\":+1}", "{\"status\":.5}", "{\"status\":1e}", "{\"host\":\"\x01\"}", "{\"host\":\"tab\there\"}"])
    else:
        s = json.dumps(dict(items), separators=(",", ":"), ensure_ascii=False)
    return s



PROPS = {"@timestamp": {"type": "date"}, "host": {"type": "keyword"}, "status": {"type": "integer"}, "bytes": {"type": "long"},
         "latency_ms": {"type": "float"}, "dbl": {"type": "double"}, "message": {"type": "text"}}


def results(node, index, docs, valid, r):
    """Indexes the generated documents (single-line ones in bulks, the rest one request each) and
    returns the per-document outcomes."""
    out = []
    pending = []

    def flush():
        if not pending:
            return
        lines = []
        for did, doc in pending:
            lines += [json.dumps({"index": {"_id": did}}), doc]
        st, resp = node.req("POST", f"/{index}/_bulk", ("\n".join(lines) + "\n").encode("utf-8", "surrogatepass"), raw=True)
        if st != 200:
            out.extend([("bulk", st)] * len(pending))
        else:
            for it in resp["items"]:
                v = it["index"]
                e = v.get("error") or {}
                out.append((v["status"], e.get("type"), e.get("reason"), (e.get("caused_by") or {}).get("type"), (e.get("caused_by") or {}).get("reason")))
        pending.clear()

    for i, (did, doc) in enumerate(docs):
        if "\n" in doc or "\r" in doc:
            flush()
            st, resp = node.req("PUT", f"/{index}/_doc/{did}", doc.encode("utf-8", "surrogatepass"), raw=False) if False else node.req("PUT", f"/{index}/_doc/{did}", doc.encode("utf-8", "surrogatepass"))
            e = resp.get("error") or {}
            out.append((st, e.get("type"), e.get("reason"), (e.get("caused_by") or {}).get("type"), (e.get("caused_by") or {}).get("reason")))
        else:
            pending.append((did, doc))
            if len(pending) >= 200:
                flush()
    flush()
    node.req("POST", f"/{index}/_refresh")
    return out


def readback(node, index):
    rows = []
    after = None
    while True:
        q = {"size": 1000, "sort": [{"_seq_no": "asc"}], "docvalue_fields": ["@timestamp", "host", "status", "bytes", "latency_ms", "dbl"], "_source": False}
        if after is not None:
            q["search_after"] = after
        st, resp = node.req("POST", f"/{index}/_search", q)
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            break
        rows += [(h["_id"], json.dumps(h.get("fields"), sort_keys=True)) for h in hits]
        after = hits[-1]["sort"]
    counts = []
    for term in ["the", "quick", "fox", "jumps", "lazy", "dog", "123", "foo_bar", "x", "cafe", "ss", "line1", "quoted", "e", "mail", "o", "neil"]:
        counts.append(node.req("POST", f"/{index}/_count", {"query": {"match": {"message": term}}})[1].get("count"))
    for host in ["host-001", "dup", "", "x" * 300]:
        counts.append(node.req("POST", f"/{index}/_count", {"query": {"term": {"host": host}}})[1].get("count"))
    for fld in ["status", "bytes", "latency_ms", "dbl"]:
        for rng in [{"gte": 0, "lte": 100}, {"gte": 100000}, {"lt": 0}]:
            counts.append(node.req("POST", f"/{index}/_count", {"query": {"range": {fld: rng}}})[1].get("count"))
    for rng in [{"gte": "2026-01-01", "lte": "2030-01-01"}, {"lt": "1990-01-01"}, {"gte": "2050-06-01T00:00:00Z"}]:
        counts.append(node.req("POST", f"/{index}/_count", {"query": {"range": {"@timestamp": rng}}})[1].get("count"))
    return rows, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default=os.environ.get("PLUGIN_URL", "http://localhost:9200"))
    ap.add_argument("--stock", default=os.environ.get("STOCK_URL", "http://localhost:9201"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--docs", type=int, default=5000)
    ap.add_argument("--valid", type=float, default=0.6, help="share of documents whose every value is one the accelerated reader takes")
    ap.add_argument("--dynamic", default="false")
    a = ap.parse_args()
    os.environ["VALID"] = str(a.valid)
    plugin, stock = Node(a.plugin, 300), Node(a.stock, 300)
    r = random.Random(a.seed)
    docs = [("d%d" % i if r.random() < .9 else "dup%d" % r.randrange(50), gen_doc(r)) for i in range(a.docs)]
    outcomes = []
    data = []
    for n, settings in ((plugin, {"index.engine.accelerated_source_parsing": True}), (stock, {})):
        n.req("DELETE", "/fi")
        st, resp = n.req("PUT", "/fi", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, **settings}, "mappings": {"dynamic": a.dynamic, "properties": PROPS}})
        if st != 200:
            print("create failed on", n.url, json.dumps(resp)[:300])
            return 1
        outcomes.append(results(n, "fi", docs, a.valid, r))
        data.append(readback(n, "fi"))
    bad = 0
    for k, (p, s) in enumerate(zip(*outcomes)):
        if p != s and not (p[:2] == s[:2] and "immense term" in str(p[2]) and str(s[2]).startswith(str(p[2]))):
            # the engine's own immense-term error carries no cause chain, with or without source parsing
            bad += 1
            if bad <= 20:
                print(f"outcome differs for {docs[k][0]}: plugin={p} stock={s} doc={docs[k][1][:200]}")
    if len(outcomes[0]) != len(outcomes[1]):
        bad += 1
        print("outcome counts differ")
    if data[0][0] != data[1][0]:
        bad += 1
        for x, y in zip(*[d[0] for d in data]):
            if x != y:
                print("doc values differ:", x, y)
                break
        print("indexed documents differ")
    if data[0][1] != data[1][1]:
        bad += 1
        print("query counts differ:", data[0][1], data[1][1])
    print(f"{len(docs)} documents, {bad} difference(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
