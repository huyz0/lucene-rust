"""Shape sets for wiki_bench.py (BENCH_EXT=confluence_shapes, KIND=conf|pdf): Confluence-style wiki search and PDF-style long documents,
both built from the Wikipedia corpus that `wiki_bench.py prep` wrote.

  KIND=conf  one doc per article: title, text (body), space (keyword), labels (keywords), perms (keywords: "everyone" or groups),
             updated (date), n. Searches as a wiki search box does it: multi_match title^3,text with a permission filter and a space
             filter, highlight, facets, sort by date, title-prefix, phrase slop, rescore, combined_fields, function_score, span, fuzzy.
  KIND=pdf   articles concatenated into documents of 5-100 pages (3000 chars a page, the text a PDF extractor yields): filename,
             content, pages, author, created. Searches: ranked OR / phrase / slop / query_string over long text, highlight, filters.

  BENCH_EXT=confluence_shapes KIND=conf python3 -I wiki_bench.py prep /data/wiki /data/conf     (reads /data/wiki/{bulk-*,queries.json})
  then wiki-ab.sh with OUT_DIR=/data/conf BENCH_EXT=confluence_shapes KIND=conf TAG=conf
"""
import glob
import json
import os
import random
import re

KIND = os.environ.get("KIND", "conf")
INDEX = KIND
SIZE = int(os.environ.get("SIZE", "10"))
NQ = int(os.environ.get("NQ", "60"))
TOK = re.compile(r"[a-z0-9]+")

if KIND == "conf":
    MAPPING = {"properties": {"title": {"type": "text"}, "text": {"type": "text"}, "n": {"type": "long"}, "space": {"type": "keyword"},
                              "labels": {"type": "keyword"}, "perms": {"type": "keyword"}, "updated": {"type": "date"}}}
else:
    MAPPING = {"properties": {"filename": {"type": "keyword"}, "content": {"type": "text"}, "pages": {"type": "integer"},
                              "author": {"type": "keyword"}, "created": {"type": "date"}}}

DAY0 = 16436  # 2015-01-01 in days since the epoch


def _date(n):
    import datetime
    return (datetime.date(1970, 1, 1) + datetime.timedelta(days=DAY0 + (n * 7919) % 3650)).isoformat()


def prep(a):
    src, out = a[0], a[1]
    os.makedirs(out, exist_ok=True)
    rnd = random.Random(5)
    WQ = json.load(open(os.path.join(src, "queries.json")))
    arts = []
    for f in sorted(glob.glob(os.path.join(src, "bulk-*.ndjson"))):
        L = open(f).read().split("\n")
        for i in range(1, len(L) - 1, 2):
            arts.append(json.loads(L[i]))
    batch, files = [], []

    def flush():
        p = os.path.join(out, "bulk-%05d.ndjson" % len(files))
        open(p, "w").write("\n".join(batch) + "\n")
        files.append(p)
        batch.clear()

    if KIND == "conf":
        for d in arts:
            n = d["n"]
            perms = ["everyone"] if n % 5 else ["g%d" % (n % 30), "g%d" % (40 + (n // 30) % 11)]
            batch.append('{"index":{"_id":"%d"}}' % n)
            batch.append(json.dumps({"title": d["title"], "text": d["text"], "n": n, "space": "S%02d" % (n % 50),
                                     "labels": ["l%d" % (n % 13), "l%d" % (20 + n % 7)], "perms": perms, "updated": _date(n)},
                                    separators=(",", ":")))
            if len(batch) >= 1000:
                flush()
        if batch:
            flush()
    else:
        i, k, pdfs = 0, 0, 0
        while i < len(arts):
            pages = rnd.choice([5, 8, 12, 20, 30, 45, 60, 80, 100])
            need, parts = pages * 3000, []
            while i < len(arts) and need > 0:
                parts.append(arts[i]["text"])
                need -= len(arts[i]["text"])
                i += 1
            batch.append('{"index":{"_id":"%d"}}' % k)
            batch.append(json.dumps({"filename": "doc-%05d.pdf" % k, "content": "\n\f".join(parts), "pages": pages,
                                     "author": "a%d" % (k % 40), "created": _date(k)}, separators=(",", ":")))
            k += 1
            if len(batch) >= 40:  # 20 PDFs (up to 300 KB each) per bulk
                flush()
        if batch:
            flush()
        print(f"{k} pdf docs")

    def pool(name):
        return WQ[name]

    frag = pool("fragments 6 words (OR)")
    Q = {}
    if KIND == "conf":
        mixed = pool("match frequent+mid (OR)") + pool("match 2 mid (OR)") + pool("match 4 mixed (OR)")
        rnd.shuffle(mixed)
        mixed = mixed[:NQ * 2]
        ph3, ph2 = pool("phrase 3 words (real)"), pool("phrase 2 words (real)")
        titles = pool("titles on text (OR)")
        Q["cf search: multi_match + perms + space filter"] = mixed
        Q["cf search + highlight (text)"] = mixed
        Q["cf search + facets (space, labels)"] = mixed
        Q["cf search sorted by updated"] = mixed
        Q["cf title prefix (match_phrase_prefix)"] = [_prefix(t, rnd) for t in titles]
        Q["cf phrase slop 2"] = ph3
        Q["cf phrase slop 5"] = ph3
        Q["cf phrase + rescore (phrase slop 3)"] = frag
        Q["cf combined_fields title^3,text"] = mixed
        Q["cf function_score (gauss updated + label boost)"] = mixed
        Q["cf span_near slop 3 in order"] = ph2
        Q["cf fuzzy AUTO"] = [_typo(q, rnd) for q in pool("match 2 mid (OR)")]
        Q["cf query_string phrase AND term"] = [json.dumps([p, m.split()[0]]) for p, m in zip(ph2 * 3, mixed)]
        Q["cf bool filters only (perms+space+label, size 20)"] = [json.dumps([rnd.randrange(50), rnd.randrange(13)]) for _ in range(NQ)]
    else:
        Q["pdf ranked OR (6-word fragments)"] = frag
        Q["pdf ranked AND (6-word fragments)"] = frag
        Q["pdf match 2 mid (OR)"] = pool("match 2 mid (OR)")
        Q["pdf phrase 2 words"] = pool("phrase 2 words (real)")
        Q["pdf phrase 3 words slop 2"] = pool("phrase 3 words (real)")
        Q["pdf fragments + highlight (3 x 150)"] = frag
        Q["pdf fragments + full _source (10 long docs)"] = frag
        Q["pdf query_string phrase AND term"] = [json.dumps([p, m.split()[0]]) for p, m in zip(pool("phrase 2 words (real)") * 3, pool("match 2 mid (OR)") * 3)]
        Q["pdf fragments + author filter + pages range"] = frag
        Q["pdf phrase + rescore (slop 3)"] = frag
    json.dump(Q, open(os.path.join(out, "queries.json"), "w"), indent=1)
    print(f"{len(arts)} articles -> {len(files)} bulk files, {len(Q)} shapes")


def _prefix(t, rnd):
    w = TOK.findall(t.lower())
    if len(w) == 1:
        return w[0][:max(2, len(w[0]) - 2)]
    return w[0] + " " + w[1][:max(1, len(w[1]) // 2)]


def _typo(q, rnd):
    out = []
    for w in q.split():
        if len(w) > 4 and rnd.random() < 0.7:
            i = rnd.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


USER_GROUPS = ["everyone"] + ["g%d" % i for i in range(0, 30, 5)] + ["g%d" % (40 + i) for i in (1, 4, 7)]


def _conf_filters(extra=None):
    f = [{"terms": {"perms": USER_GROUPS}}, {"term": {"space": "S07"}} if False else {"terms": {"space": ["S%02d" % i for i in range(0, 50, 3)]}}]
    return f + (extra or [])


def body_for(name, q):
    if name.startswith("cf "):
        return _conf_body(name, q)
    return _pdf_body(name, q)


def _conf_body(name, q):
    base = {"size": SIZE}
    mm = {"multi_match": {"query": q, "fields": ["title^3", "text"]}}
    if name.startswith("cf search: "):
        base["query"] = {"bool": {"must": mm, "filter": _conf_filters()}}
    elif name.startswith("cf search + highlight"):
        base["query"] = {"bool": {"must": mm, "filter": _conf_filters()}}
        base["highlight"] = {"fields": {"text": {"fragment_size": 150, "number_of_fragments": 3}, "title": {"number_of_fragments": 0}},
                             "pre_tags": ["<em>"], "post_tags": ["</em>"]}
    elif name.startswith("cf search + facets"):
        base["query"] = {"bool": {"must": mm, "filter": _conf_filters()}}
        base["aggs"] = {"spaces": {"terms": {"field": "space", "size": 10}}, "labels": {"terms": {"field": "labels", "size": 10}}}
    elif name.startswith("cf search sorted"):
        base["query"] = {"bool": {"must": mm, "filter": _conf_filters()}}
        base["sort"] = [{"updated": "desc"}, "_score"]
    elif name.startswith("cf title prefix"):
        base["query"] = {"bool": {"must": {"match_phrase_prefix": {"title": {"query": q, "max_expansions": 50}}}, "filter": _conf_filters()}}
    elif name.startswith("cf phrase slop 2"):
        base["query"] = {"bool": {"must": {"match_phrase": {"text": {"query": q, "slop": 2}}}, "filter": _conf_filters()}}
    elif name.startswith("cf phrase slop 5"):
        base["query"] = {"bool": {"must": {"match_phrase": {"text": {"query": q, "slop": 5}}}, "filter": _conf_filters()}}
    elif name.startswith("cf phrase + rescore"):
        base["query"] = {"bool": {"must": {"match": {"text": q}}, "filter": _conf_filters()}}
        base["rescore"] = {"window_size": 100, "query": {"rescore_query": {"match_phrase": {"text": {"query": q, "slop": 3}}},
                                                         "query_weight": 1.0, "rescore_query_weight": 2.0}}
    elif name.startswith("cf combined_fields"):
        base["query"] = {"bool": {"must": {"combined_fields": {"query": q, "fields": ["title^3", "text"]}}, "filter": _conf_filters()}}
    elif name.startswith("cf function_score"):
        base["query"] = {"function_score": {"query": {"bool": {"must": mm, "filter": _conf_filters()}},
                                            "functions": [{"gauss": {"updated": {"origin": "2024-12-31", "scale": "365d", "decay": 0.5}}},
                                                          {"filter": {"term": {"labels": "l3"}}, "weight": 1.5}],
                                            "score_mode": "multiply", "boost_mode": "multiply"}}
    elif name.startswith("cf span_near"):
        w = q.split()
        base["query"] = {"bool": {"must": {"span_near": {"clauses": [{"span_term": {"text": x}} for x in w], "slop": 3, "in_order": True}},
                                  "filter": _conf_filters()}}
    elif name.startswith("cf fuzzy"):
        base["query"] = {"bool": {"must": {"match": {"text": {"query": q, "fuzziness": "AUTO"}}}, "filter": _conf_filters()}}
    elif name.startswith("cf query_string"):
        p, t = json.loads(q)
        base["query"] = {"bool": {"must": {"query_string": {"query": f'"{p}" AND {t}', "fields": ["title^3", "text"]}}, "filter": _conf_filters()}}
    elif name.startswith("cf bool filters only"):
        s, l = json.loads(q)
        base["size"] = 20
        base["query"] = {"bool": {"filter": _conf_filters([{"term": {"space": "S%02d" % s}}, {"term": {"labels": "l%d" % l}}])}}
        base["sort"] = [{"updated": "desc"}]
    else:
        raise KeyError(name)
    return base


def _pdf_body(name, q):
    base = {"size": SIZE}
    if name.startswith("pdf ranked AND"):
        base["query"] = {"match": {"content": {"query": q, "operator": "and"}}}
    elif name.startswith("pdf phrase 2"):
        base["query"] = {"match_phrase": {"content": q}}
    elif name.startswith("pdf phrase 3"):
        base["query"] = {"match_phrase": {"content": {"query": q, "slop": 2}}}
    elif name.startswith("pdf query_string"):
        p, t = json.loads(q)
        base["query"] = {"query_string": {"query": f'"{p}" AND {t}', "fields": ["content"]}}
    elif name.startswith("pdf fragments + author"):
        base["query"] = {"bool": {"must": {"match": {"content": q}}, "filter": [{"terms": {"author": ["a%d" % i for i in range(0, 40, 2)]}},
                                                                                 {"range": {"pages": {"gte": 10, "lte": 80}}}]}}
    elif name.startswith("pdf phrase + rescore"):
        base["query"] = {"match": {"content": q}}
        base["rescore"] = {"window_size": 50, "query": {"rescore_query": {"match_phrase": {"content": {"query": q, "slop": 3}}},
                                                        "query_weight": 1.0, "rescore_query_weight": 2.0}}
    else:
        base["query"] = {"match": {"content": q}}
    if "highlight" in name:
        base["highlight"] = {"fields": {"content": {"fragment_size": 150, "number_of_fragments": 3}}}
    if "full _source" not in name:
        base["_source"] = ["filename", "pages"]
    return base
