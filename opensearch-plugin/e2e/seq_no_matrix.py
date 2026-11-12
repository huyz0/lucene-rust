"""Lean sequence-number layouts (index.lucene_rust.seq_no.*) against stock OpenSearch.

  python3 seq_no_matrix.py STOCK_URL PLUGIN_URL

Indexes the same operations (one client, so both nodes assign the same sequence numbers) into a
stock index and into plugin indices with the default layout, `doc_values_only`, and `doc_values_only`
plus `disable_sequence_numbers`, and `doc_values_only` under realtime search (searched while the
documents are still in the writer's buffers), then compares searches on `_seq_no`, fetched sequence numbers,
gets, nested and text searches, before and after a force merge and across a close/open. The
other layouts must answer exactly as stock; with `disable_sequence_numbers` the documents the
merge left without history report no sequence number, and everything else still matches.
"""
import json, sys, time, urllib.error, urllib.request

STOCK, PLUGIN = sys.argv[1], sys.argv[2]
N = 6000

def req(base, method, path, body=None, ndjson=False, ok=(200, 201)):
    data, headers = None, {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

def ok(base, method, path, body=None, ndjson=False):
    status, r = req(base, method, path, body, ndjson)
    if status not in (200, 201) or (isinstance(r, dict) and r.get("errors")):
        raise RuntimeError(f"{method} {path}: {status} {json.dumps(r)[:600]}")
    return r

VARIANTS = {
    "default": {},
    "doc_values_only": {"lucene_rust.seq_no.index_options": "doc_values_only"},
    "disable_sequence_numbers": {"lucene_rust.seq_no.index_options": "doc_values_only",
                                 "lucene_rust.disable_sequence_numbers": True},
    # Searched before any refresh: the documents are still in the writer's buffers.
    "doc_values_only_realtime": {"lucene_rust.seq_no.index_options": "doc_values_only",
                                 "lucene_rust.realtime_search": True, "lucene_rust.realtime_max_staleness": "0ms"},
}
MAPPING = {"properties": {"n": {"type": "long"}, "msg": {"type": "text"},
                          "tags": {"type": "nested", "properties": {"k": {"type": "keyword"}}}}}

def load(base, index, extra):
    req(base, "DELETE", "/" + index)
    settings = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1,
                **extra}
    ok(base, "PUT", "/" + index, {"settings": {"index": settings}, "mappings": MAPPING})
    for b in range(0, N, 1500):
        lines = []
        for i in range(b, b + 1500):
            doc = {"n": i, "msg": f"event {i % 17} value {i}"}
            if i % 5 == 0:
                doc["tags"] = [{"k": f"t{i % 3}"}, {"k": "all"}]
            lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps(doc)]
        ok(base, "POST", f"/{index}/_bulk", ("\n".join(lines) + "\n").encode(), ndjson=True)
    lines = []
    for i in range(0, N, 30):
        lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps({"n": i, "msg": f"updated {i}"})]
    for i in range(7, N, 61):
        lines.append(json.dumps({"delete": {"_id": f"d{i}"}}))
    ok(base, "POST", f"/{index}/_bulk", ("\n".join(lines) + "\n").encode(), ndjson=True)
    if not extra.get("lucene_rust.realtime_search"):
        ok(base, "POST", f"/{index}/_refresh")

SEARCHES = {
    "range": {"query": {"range": {"_seq_no": {"gte": 1000, "lt": 1400}}}, "size": 5, "sort": [{"_seq_no": "asc"}], "track_total_hits": True},
    "range_open": {"query": {"range": {"_seq_no": {"gt": 5900}}}, "size": 0, "track_total_hits": True},
    "term": {"query": {"term": {"_seq_no": 4321}}},
    "terms": {"query": {"terms": {"_seq_no": [3, 99, 2500, 6100, 99999]}}, "sort": [{"_seq_no": "asc"}]},
    "bool": {"query": {"bool": {"must": [{"match": {"msg": "event"}}], "filter": [{"range": {"_seq_no": {"lte": 800}}}]}},
             "size": 3, "track_total_hits": True},
    "sort_seq_no": {"query": {"match_all": {}}, "sort": [{"_seq_no": "desc"}], "size": 20},
    "fetch": {"query": {"match": {"msg": "updated"}}, "sort": [{"n": "asc"}], "size": 25, "seq_no_primary_term": True},
    "nested": {"query": {"nested": {"path": "tags", "query": {"term": {"tags.k": "t1"}}}}, "size": 0, "track_total_hits": True},
    "text": {"query": {"match": {"msg": "value"}}, "size": 0, "track_total_hits": True},
    "agg_seq_no": {"size": 0, "aggs": {"max": {"max": {"field": "_seq_no"}}, "min": {"min": {"field": "_seq_no"}}}},
}
SEQ_ONLY = {"range", "range_open", "term", "terms", "bool", "sort_seq_no", "agg_seq_no"}

def norm(r):
    if not isinstance(r, dict) or "hits" not in r:
        return r
    h = r["hits"]
    out = {"total": h.get("total"), "hits": [{k: x.get(k) for k in ("_id", "_seq_no", "_primary_term", "sort")} for x in h["hits"]]}
    if "aggregations" in r:
        out["aggs"] = r["aggregations"]
    return out

def answers(base, index):
    a = {name: norm(ok(base, "POST", f"/{index}/_search", body)) for name, body in SEARCHES.items()}
    for doc in ("d1", "d30", "d4999"):
        status, g = req(base, "GET", f"/{index}/_doc/{doc}")
        a["get_" + doc] = {k: g.get(k) for k in ("found", "_seq_no", "_primary_term", "_version")}
    return a

def merge(base, index):
    ok(base, "POST", f"/{index}/_flush")
    time.sleep(35)  # the shard renews its retention lease to the global checkpoint every 30 s
    ok(base, "POST", f"/{index}/_forcemerge?max_num_segments=1")
    ok(base, "POST", f"/{index}/_refresh")

failures = []
def compare(stage, variant, expected, got, skip=()):
    for k in expected:
        if k in skip:
            continue
        if expected[k] != got.get(k):
            failures.append(f"{variant} {stage} {k}:\n  stock  {json.dumps(expected[k])[:400]}\n  plugin {json.dumps(got.get(k))[:400]}")

load(STOCK, "seqno", {})
stock_before = answers(STOCK, "seqno")
merge(STOCK, "seqno")
stock_after = answers(STOCK, "seqno")
if stock_before != stock_after:
    raise SystemExit("stock answers changed across a force merge")

for variant, extra in VARIANTS.items():
    index = "seqno-" + variant.replace("_", "-")
    load(PLUGIN, index, extra)
    compare("before merge", variant, stock_before, answers(PLUGIN, index))
    merge(PLUGIN, index)
    after = answers(PLUGIN, index)
    if variant != "disable_sequence_numbers":
        compare("after merge", variant, stock_after, after)
    else:
        compare("after merge", variant, stock_after, after, skip=SEQ_ONLY | {"fetch", "get_d1", "get_d30", "get_d4999"})
        for doc in ("d1", "d30", "d4999"):
            g = after["get_" + doc]
            if g["found"] != True or g["_seq_no"] is not None or g["_primary_term"] is not None:
                failures.append(f"{variant}: get {doc} after merge: {g}")
        if any(x["_seq_no"] is not None for x in after["fetch"]["hits"]) or after["fetch"]["total"] != stock_after["fetch"]["total"]:
            failures.append(f"{variant}: fetch after merge reports sequence numbers: {after['fetch']['hits'][:3]}")
        if after["range"]["total"]["value"] != 0 or after["agg_seq_no"]["aggs"]["max"]["value"] != -2:
            failures.append(f"{variant}: _seq_no after merge not the sentinel: {after['range']['total']} {after['agg_seq_no']}")
        # Writes on documents without history.
        status, r = req(PLUGIN, "PUT", f"/{index}/_doc/d2?if_seq_no=2&if_primary_term=1", {"n": 2, "msg": "cas"})
        if status != 409:
            failures.append(f"{variant}: if_seq_no on a sentinel document answered {status} {r}")
        status, r = req(PLUGIN, "POST", f"/{index}/_update/d3", {"doc": {"msg": "partial"}})
        if status != 200 or r.get("result") != "updated":
            failures.append(f"{variant}: update of a sentinel document answered {status} {r}")
        status, r = req(PLUGIN, "PUT", f"/{index}/_doc/d4", {"n": 4, "msg": "overwrite"})
        if status != 200 or r.get("result") != "updated" or r.get("_seq_no", -1) < N:
            failures.append(f"{variant}: overwrite of a sentinel document answered {status} {r}")
        ok(PLUGIN, "POST", f"/{index}/_refresh")
        status, g = req(PLUGIN, "GET", f"/{index}/_doc/d4")
        if g.get("_seq_no") != r.get("_seq_no") or g["_source"]["msg"] != "overwrite":
            failures.append(f"{variant}: get after overwrite {g}")
    # Close and open: the engine restores its version map and checkpoint from the index.
    ok(PLUGIN, "POST", f"/{index}/_close")
    ok(PLUGIN, "POST", f"/{index}/_open?wait_for_active_shards=1")
    ok(PLUGIN, "GET", f"/_cluster/health/{index}?wait_for_status=green&timeout=60s")
    reopened = answers(PLUGIN, index)
    if variant != "disable_sequence_numbers":
        compare("after reopen", variant, stock_after, reopened)
    else:
        # The update and the overwrite above took "value" out of two messages.
        expect = {"text": stock_after["text"]["total"]["value"] - 2, "nested": stock_after["nested"]["total"]["value"]}
        for k in ("text", "nested"):
            if reopened[k]["total"]["value"] != expect[k]:
                failures.append(f"{variant}: {k} after reopen {reopened[k]['total']} vs {expect[k]}")
    status, r = req(PLUGIN, "PUT", f"/{index}/_doc/new1", {"n": 1, "msg": "after reopen"})
    if status not in (200, 201):
        failures.append(f"{variant}: write after reopen {status} {r}")
    print(f"{variant}: done")

if failures:
    print("\n".join(failures))
    raise SystemExit(f"{len(failures)} difference(s)")
print("seq_no matrix: every layout answers as stock")
