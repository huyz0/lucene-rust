"""Highlighting edge cases on the plugin node against stock OpenSearch, beyond
highlight_matrix.py: requests OpenSearch's highlighters refuse (the plugin must
fail with the same error, from the native path), text with unpaired surrogates,
the analyzed-offset limit, derived fields, and the Thai break iterator.

Usage: highlight_extra.py [load|check|all]   (PLUGIN, STOCK as in highlight_matrix.py)
A row prints a difference, or nothing; the exit status is the number of differences. The
plugin's stats then show the highlight fallbacks: none of the cases here may add one."""
import json
import os
import random
import sys

import requests

import highlight_matrix as hm

sys.stdout.reconfigure(errors="backslashreplace")

PLUGIN, STOCK, S = hm.PLUGIN, hm.STOCK, hm.S
TV = "with_positions_offsets"
LONE = ["\ud83d", "\ude00", "\ud800", "\udfff"]


def body_with_surrogates(rnd):
    words = hm.text(rnd, 1, 3).split(" ")
    out = []
    for w in words:
        out.append(w)
        r = rnd.random()
        if r < 0.15:
            out.append(rnd.choice(LONE))  # a lone surrogate as a token of its own
        elif r < 0.3:
            out[-1] = w + rnd.choice(LONE)  # glued to a word
        elif r < 0.35:
            out.append("😀")  # a proper pair
    return " ".join(out)


def surrogate_docs(n):
    rnd = random.Random(7)
    docs = []
    for i in range(n):
        d = {"body": body_with_surrogates(rnd), "title": hm.sentence(rnd), "num": i}
        if i % 5 == 0:
            d["body"] = d["body"] + " fox " + rnd.choice(LONE) + " dog"
        docs.append(d)
    return docs


def limit_docs(n):
    rnd = random.Random(11)
    out = []
    for i in range(n):
        body = hm.text(rnd, 1, 4) if i % 3 else hm.sentence(rnd)
        out.append({"body": body, "title": hm.sentence(rnd), "num": i, "kw": rnd.choice(hm.WORDS)})
    return out


def derived_docs(n):
    rnd = random.Random(13)
    return [{"body": hm.text(rnd, 1, 3), "title": hm.sentence(rnd), "num": i} for i in range(n)]


THAI = ["กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย", "ฉันชอบกินข้าวผัดและต้มยำกุ้ง", "การค้นหาข้อความภาษาไทยต้องตัดคำให้ถูกต้อง",
        "วันนี้อากาศดีมากเราไปเที่ยวทะเลกันเถอะ", "ระบบค้นหาข้อมูลขนาดใหญ่ทำงานได้รวดเร็ว", "สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ"]


def thai_docs(n):
    rnd = random.Random(17)
    return [{"body": " ".join(rnd.choice(THAI) for _ in range(rnd.randint(1, 5))) + " fox", "title": rnd.choice(THAI), "num": i}
            for i in range(n)]


def text_mapping(**kw):
    return {"properties": {"body": {"type": "text", "term_vector": TV, **kw}, "title": {"type": "text"},
                           "num": {"type": "integer"}, "kw": {"type": "keyword"}}}


INDEXES = {
    # name: (settings, mapping, docs)
    "hlx_surrogate": ({}, text_mapping(), surrogate_docs(300)),
    "hlx_surrogate_ws": ({"analysis": hm.ANALYSIS}, text_mapping(analyzer="ws_lower"), surrogate_docs(300)),
    "hlx_limit": ({"highlight.max_analyzed_offset": 60}, text_mapping(), limit_docs(120)),
    "hlx_thai": ({}, text_mapping(), thai_docs(120)),
    # the plugin's second half stays in the writer's buffers, unrefreshed
    "hlx_live": ({}, {"properties": {"body": {"type": "text", "index_options": "offsets"}, "title": {"type": "text"},
                                    "num": {"type": "integer"}, "kw": {"type": "keyword"}}}, limit_docs(120)),
    "hlx_derived": ({}, {
        "properties": {"body": {"type": "text"}, "title": {"type": "text"}, "num": {"type": "integer"}},
        "derived": {
            "dtitle": {"type": "keyword", "script": {"source": "emit(params._source.title)"}},
            "dnum": {"type": "long", "script": {"source": "emit(params._source.num)"}},
        },
    }, derived_docs(120)),
}


LIVE = {"hlx_live"}
LIVE_SETTINGS = {"engine.accelerated": True, "search.realtime.enabled": True, "search.realtime.max_staleness": "0ms", "refresh_interval": -1}


def bulk(base, name, docs, start):
    lines = []
    for i, d in enumerate(docs, start):
        lines.append(json.dumps({"index": {"_id": str(i)}}))
        lines.append(json.dumps(d))
    r = S.post(f"{base}/{name}/_bulk", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"})
    assert r.ok and not r.json()["errors"], r.text[:500]


def load():
    for base in (PLUGIN, STOCK):
        for name, (settings, mapping, docs) in INDEXES.items():
            S.delete(f"{base}/{name}")
            live = name in LIVE
            extra = LIVE_SETTINGS if live and base == PLUGIN else {"refresh_interval": -1} if live else {}
            body = {"settings": {"number_of_shards": 1, "number_of_replicas": 0, **settings, **extra}, "mappings": mapping}
            r = S.put(f"{base}/{name}", json=body)
            assert r.ok, r.text
            half = len(docs) // 2 if live else len(docs)
            bulk(base, name, docs[:half], 0)
            S.post(f"{base}/{name}/_refresh")
            if live:
                bulk(base, name, docs[half:], half)
                if base == STOCK:
                    S.post(f"{base}/{name}/_refresh")
    print("loaded", len(INDEXES), "indices on both nodes")


Q_BODY = {"match": {"body": "fox dog"}}
Q_PHRASE = {"match_phrase": {"body": "quick fox"}}
Q_ALL = {"match_all": {}}
Q_PREFIX = {"prefix": {"body": "fo"}}

HL_TYPES = ["unified", "plain", "fvh"]


def per_type(options):
    return {f"{t}:{n}": ({**o, "type": t} if "type" not in o else o) for t in HL_TYPES for n, o in options.items()}


COMMON = {
    "default": {"fields": {"body": {}}},
    "frags0": {"fields": {"body": {"number_of_fragments": 0}}},
    "size20": {"fields": {"body": {"fragment_size": 20, "number_of_fragments": 3}}},
    "no_match": {"fields": {"body": {"no_match_size": 30}}},
    "html": {"encoder": "html", "fields": {"body": {}}},
    "word": {"fields": {"body": {"boundary_scanner": "word", "fragment_size": 25}}},
    "sentence": {"fields": {"body": {"boundary_scanner": "sentence", "fragment_size": 40}}},
}

ERRORS = {
    "fvh_offset_negative": {"type": "fvh", "fields": {"body": {"fragment_offset": -3}}},
    "fvh_offset_zero_small": {"type": "fvh", "fields": {"body": {"fragment_offset": 0, "fragment_size": 5}}},
    "fvh_small_size": {"type": "fvh", "fields": {"body": {"fragment_size": 10}}},
    "fvh_size_with_offset": {"type": "fvh", "fields": {"body": {"fragment_offset": 20, "fragment_size": 50}}},
    "fvh_chars_scanner": {"type": "fvh", "fields": {"body": {"boundary_scanner": "chars", "fragment_size": 30}}},
    "unified_chars": {"type": "unified", "fields": {"body": {"boundary_scanner": "chars"}}},
    "plain_bad_fragmenter": {"type": "plain", "fields": {"body": {"fragmenter": "bogus"}}},
    "plain_offset_over": {"type": "plain", "fields": {"body": {"max_analyzer_offset": 2000000}}},
    "unified_offset_over": {"type": "unified", "fields": {"body": {"max_analyzer_offset": 2000000}}},
    "plain_offset_ok": {"type": "plain", "fields": {"body": {"max_analyzer_offset": 20}}},
    "unified_offset_ok": {"type": "unified", "fields": {"body": {"max_analyzer_offset": 20}}},
    "fvh_no_tv_title": {"type": "fvh", "fields": {"title": {}}},
}

CASES = []  # (index, query name, query, option name, options)


def add(index, qn, q, sets):
    for on, o in sets.items():
        CASES.append((index, qn, q, on, o))


for qn, q in (("body", Q_BODY), ("phrase", Q_PHRASE), ("prefix", Q_PREFIX), ("all", Q_ALL)):
    for ix in ("hlx_surrogate", "hlx_surrogate_ws"):
        add(ix, qn, q, per_type(COMMON))
    add("hlx_limit", qn, q, per_type(COMMON))
    add("hlx_derived", qn, q, {k: v for k, v in per_type(COMMON).items() if not k.startswith("fvh")})
    add("hlx_thai", qn, q, per_type(COMMON))
    add("hlx_live", qn, q, {k: v for k, v in per_type(COMMON).items() if not k.startswith("fvh")})
    for ix in ("hlx_surrogate", "hlx_limit", "hlx_thai"):
        add(ix, qn, q, ERRORS)

# derived fields: queries on them, and highlighting them
DERIVED_QUERIES = {
    "dtitle_term": {"term": {"dtitle": "fox"}},
    "dtitle_prefix": {"prefix": {"dtitle": "q"}},
    "dtitle_match_body": {"bool": {"must": [{"match": {"body": "fox"}}], "should": [{"term": {"dtitle": "dog"}}]}},
    "dnum_range": {"range": {"dnum": {"gte": 10, "lte": 40}}},
}
for qn, q in DERIVED_QUERIES.items():
    add("hlx_derived", qn, q, {
        "unified_dtitle": {"type": "unified", "fields": {"dtitle": {}, "body": {}, "title": {}}},
        "plain_dtitle": {"type": "plain", "fields": {"dtitle": {}, "body": {}, "title": {}}},
        "fvh_dtitle": {"type": "fvh", "fields": {"dtitle": {}}},
        "unified_nomatch": {"type": "unified", "fields": {"dtitle": {"no_match_size": 20}}},
    })

# the Thai break iterator
THAI_OPTS = {
    "unified_word_th": {"type": "unified", "fields": {"body": {"boundary_scanner": "word", "boundary_scanner_locale": "th", "fragment_size": 30}}},
    "unified_sent_th": {"type": "unified", "fields": {"body": {"boundary_scanner": "sentence", "boundary_scanner_locale": "th"}}},
    "unified_nomatch_th": {"type": "unified", "fields": {"body": {"no_match_size": 25, "boundary_scanner_locale": "th"}}},
    "fvh_word_th": {"type": "fvh", "fields": {"body": {"boundary_scanner": "word", "boundary_scanner_locale": "th", "fragment_size": 30}}},
    "fvh_sent_th": {"type": "fvh", "fields": {"body": {"boundary_scanner": "sentence", "boundary_scanner_locale": "th", "fragment_size": 40}}},
    "fvh_nomatch_th": {"type": "fvh", "fields": {"body": {"no_match_size": 25, "boundary_scanner": "word", "boundary_scanner_locale": "th"}}},
    "plain_nomatch": {"type": "plain", "fields": {"body": {"no_match_size": 25}}},
}
for qn, q in (("fox", {"match": {"body": "fox"}}), ("thai", {"match": {"body": "ค้นหา"}}), ("all", Q_ALL)):
    add("hlx_thai", qn, q, THAI_OPTS)


def check():
    bad = 0
    for index, qn, q, on, o in CASES:
        report = hm.compare(index, qn, q, on, o, 30)
        if report is not None:
            bad += 1
            print(report)
    stats = S.get(f"{PLUGIN}/_plugins/lucene_rust/stats").json()
    print(f"{len(CASES) - bad}/{len(CASES)} rows identical")
    print("highlight fallbacks", {k: v for k, v in stats.get("fallbacks", {}).items() if "highlight" in k})
    return bad


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("load", "all"):
        load()
    if mode in ("check", "all"):
        sys.exit(1 if check() else 0)
