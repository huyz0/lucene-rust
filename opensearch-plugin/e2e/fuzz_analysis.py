#!/usr/bin/env python3
"""Indexing-side analysis differential: random multilingual text is indexed on a plugin node and a
stock node under many analyzers; the term dictionaries (terms aggregation over fielddata text
fields, with document counts) and the term vectors (positions, offsets) must be identical.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/fuzz_analysis.py --seed 1 --docs 500
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node, diff  # noqa: E402

PIECES = [
    "The Quick brown fox", "jumped over the lazy dog's back", "café naïve résumé", "Straße GROSSE Grüße",
    "İstanbul ışık İ", "ΟΔΥΣΣΕΥΣ οδυσσεύς",
    "Москва привет мир", "日本語のテキストです", "中文分词测试句子",
    "한국어 텍스트", "ภาษาไทยทดสอบ", "مرحبا بالعالم",
    "שלום עולם", "नमस्ते दुनिया", "\U0001f600 \U0001f468‍\U0001f469‍\U0001f467 \U0001f1fa\U0001f1f8",
    "é ä ô", "3.14159 1,234,567 1.2.3 -42 +7", "user@example.com http://example.com/path?q=1&x=2#frag", "wi-fi state-of-the-art x_y_z foo.bar",
    "CamelCaseWord getHTTPResponse XML2JSON", "O'Neil don't rock'n'roll l'avion", "<b>bold</b> &amp; <i>itálic</i> &lt;tag&gt;",
    "running runs ran runner easily happiness", "flies flying flew caresses ponies", "a b c d e f g h", "ﬁne ﬂuffy Ångström",
    "zero​width nbsp\ttab\nnewline", "\u0000ctrl\u0007chars", "x" * 300, "y" * 40 + "-" + "z" * 40, "\ud83d", "ǅ ǆ Ǆ", "ΑΣ ΑΣ.",
    "Iñtërnâtiônàlizætiøn", "àáâãäåçèéêë", "ﬃ ﬀ ŉ ǰ ΐ", "1st 2nd 3rd 4th", "AT&T Q&A R&D", "tel:+1-555-0100",
]


def text(r):
    n = r.randint(1, 6)
    return (" ".join(r.choice(PIECES) for _ in range(n)) if r.random() < .8 else "".join(r.choice(PIECES) + r.choice([" ", ",", ".", "-", "/", "  ", "　"]) for _ in range(n))).replace("\ud83d", "?")


ANALYZERS = {
    "standard": {"type": "standard"},
    "std_stop": {"type": "standard", "stopwords": "_english_"},
    "std_maxlen": {"type": "standard", "max_token_length": 5},
    "simple": {"type": "simple"},
    "whitespace": {"type": "whitespace"},
    "stop": {"type": "stop"},
    "keyword": {"type": "keyword"},
    "pattern": {"type": "pattern", "pattern": "\\W+", "lowercase": True},
    "english": {"type": "english"}, "french": {"type": "french"}, "german": {"type": "german"}, "spanish": {"type": "spanish"},
    "italian": {"type": "italian"}, "portuguese": {"type": "portuguese"}, "dutch": {"type": "dutch"}, "russian": {"type": "russian"},
    "arabic": {"type": "arabic"}, "cjk": {"type": "cjk"}, "greek": {"type": "greek"}, "hindi": {"type": "hindi"}, "turkish": {"type": "turkish"},
    "thai": {"type": "thai"}, "swedish": {"type": "swedish"}, "norwegian": {"type": "norwegian"}, "danish": {"type": "danish"}, "finnish": {"type": "finnish"},
    "hungarian": {"type": "hungarian"}, "romanian": {"type": "romanian"}, "czech": {"type": "czech"}, "bulgarian": {"type": "bulgarian"}, "persian": {"type": "persian"},
    "indonesian": {"type": "indonesian"}, "irish": {"type": "irish"}, "latvian": {"type": "latvian"}, "lithuanian": {"type": "lithuanian"}, "basque": {"type": "basque"},
    "catalan": {"type": "catalan"}, "galician": {"type": "galician"}, "armenian": {"type": "armenian"}, "sorani": {"type": "sorani"}, "bengali": {"type": "bengali"}, "brazilian": {"type": "brazilian"},
    "fingerprint": {"type": "fingerprint"},
    "c_classic": {"type": "custom", "tokenizer": "classic", "filter": ["lowercase", "classic"]},
    "c_letter": {"type": "custom", "tokenizer": "letter", "filter": ["lowercase"]},
    "c_lcase_tok": {"type": "custom", "tokenizer": "lowercase"},
    "c_uax": {"type": "custom", "tokenizer": "uax_url_email", "filter": ["lowercase"]},
    "c_ngram": {"type": "custom", "tokenizer": {"type": "ngram", "min_gram": 2, "max_gram": 3}, "filter": ["lowercase"]},
    "c_edge": {"type": "custom", "tokenizer": {"type": "edge_ngram", "min_gram": 1, "max_gram": 4, "token_chars": ["letter", "digit"]}},
    "c_path": {"type": "custom", "tokenizer": {"type": "path_hierarchy", "delimiter": "/"}},
    "c_pattern_tok": {"type": "custom", "tokenizer": {"type": "pattern", "pattern": "[,.]\\s*"}},
    "c_char_group": {"type": "custom", "tokenizer": {"type": "char_group", "tokenize_on_chars": ["whitespace", "-", "\n"]}},
    "c_simple_pattern": {"type": "custom", "tokenizer": {"type": "simple_pattern", "pattern": "[a-z]{3,}"}},
    "c_simple_pattern_split": {"type": "custom", "tokenizer": {"type": "simple_pattern_split", "pattern": "[ ,]"}},
    "c_thai_tok": {"type": "custom", "tokenizer": "thai"},
    "c_fold": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "asciifolding"]},
    "c_fold_pres": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "asciifolding", "preserve_original": True}]},
    "c_porter": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "porter_stem"]},
    "c_kstem": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "kstem"]},
    "c_snow": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "snowball", "language": "German2"}]},
    "c_stem_fr": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "stemmer", "language": "light_french"}]},
    "c_stem_ru": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "stemmer", "language": "russian"}]},
    "c_wdf": {"type": "custom", "tokenizer": "whitespace", "filter": [{"type": "word_delimiter", "generate_word_parts": True, "generate_number_parts": True, "catenate_all": True, "split_on_case_change": True, "preserve_original": True}]},
    "c_wdg": {"type": "custom", "tokenizer": "whitespace", "filter": [{"type": "word_delimiter_graph", "catenate_words": True, "split_on_numerics": True, "stem_english_possessive": True}]},
    "c_shingle": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "shingle", "min_shingle_size": 2, "max_shingle_size": 3, "output_unigrams": True}]},
    "c_ngram_f": {"type": "custom", "tokenizer": "keyword", "filter": [{"type": "ngram", "min_gram": 2, "max_gram": 4}]},
    "c_edge_f": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "edge_ngram", "min_gram": 2, "max_gram": 5, "preserve_original": True}]},
    "c_len": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "length", "min": 3, "max": 8}]},
    "c_trunc": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "truncate", "length": 4}]},
    "c_rev": {"type": "custom", "tokenizer": "standard", "filter": ["reverse"]},
    "c_unique": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "unique", "only_on_same_position": False}]},
    "c_elision": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "elision", "articles": ["l", "d", "qu"], "articles_case": True}, "lowercase"]},
    "c_stop_ws": {"type": "custom", "tokenizer": "whitespace", "filter": [{"type": "stop", "stopwords": ["the", "a", "of"], "ignore_case": True}]},
    "c_keep_types": {"type": "custom", "tokenizer": "uax_url_email", "filter": [{"type": "keep_types", "types": ["<EMAIL>", "<URL>"]}]},
    "c_cjk_bigram": {"type": "custom", "tokenizer": "standard", "filter": ["cjk_width", "lowercase", "cjk_bigram"]},
    "c_icu_like": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "decimal_digit", "arabic_normalization", "persian_normalization", "german_normalization", "hindi_normalization", "indic_normalization", "scandinavian_normalization", "scandinavian_folding", "sorani_normalization", "serbian_normalization"]},
    "c_syn": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "synonym", "synonyms": ["quick, fast => speedy", "dog, hound", "new york, ny"]}]},
    "c_syng": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "synonym_graph", "synonyms": ["quick, fast", "lazy dog => sloth"]}]},
    "c_cond": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "condition", "filter": ["uppercase"], "script": {"source": "token.getTerm().length() < 5"}}]},
    "c_pred": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "predicate_token_filter", "script": {"source": "token.getPosition() < 3"}}]},
    "c_limit": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "limit", "max_token_count": 3}]},
    "c_pat_cap": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "pattern_capture", "patterns": ["(\\p{Ll}+|\\d+)"], "preserve_original": True}]},
    "c_pat_rep": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "pattern_replace", "pattern": "(\\d)", "replacement": "#"}]},
    "c_html": {"type": "custom", "char_filter": ["html_strip"], "tokenizer": "standard", "filter": ["lowercase"]},
    "c_map": {"type": "custom", "char_filter": [{"type": "mapping", "mappings": ["é => e", "ph => f", "- => _"]}], "tokenizer": "standard"},
    "c_cf_pat": {"type": "custom", "char_filter": [{"type": "pattern_replace", "pattern": "(\\d+)-(?=\\d)", "replacement": "$1_"}], "tokenizer": "standard"},
    "c_keyword_marker": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "keyword_marker", "keywords": ["running", "flies"]}, "porter_stem"]},
    "c_stem_override": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", {"type": "stemmer_override", "rules": ["running => run", "flies, flying => fly"]}, "porter_stem"]},
    "c_trim": {"type": "custom", "tokenizer": "keyword", "filter": ["trim", "lowercase"]},
    "c_uppercase": {"type": "custom", "tokenizer": "standard", "filter": ["uppercase"]},
    "c_remove_dup": {"type": "custom", "tokenizer": "whitespace", "filter": ["remove_duplicates", "lowercase"]},
    "c_fp": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "asciifolding", {"type": "fingerprint", "max_output_size": 100, "separator": "_"}]},
    "c_hyph": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "flatten_graph"]},
    "c_min_hash": {"type": "custom", "tokenizer": "standard", "filter": [{"type": "min_hash", "hash_count": 2, "bucket_count": 4, "hash_set_size": 1}]},
}


def settings(shards):
    analysis = {"analyzer": {}, "tokenizer": {}, "filter": {}, "char_filter": {}}
    props = {}
    for name, spec in ANALYZERS.items():
        spec = json.loads(json.dumps(spec))
        if spec["type"] == "custom":
            tok = spec["tokenizer"]
            if isinstance(tok, dict):
                analysis["tokenizer"][name + "_t"] = tok
                spec["tokenizer"] = name + "_t"
            fl = []
            for i, f in enumerate(spec.get("filter", [])):
                if isinstance(f, dict):
                    analysis["filter"][f"{name}_f{i}"] = f
                    fl.append(f"{name}_f{i}")
                else:
                    fl.append(f)
            if fl:
                spec["filter"] = fl
            cl = []
            for i, f in enumerate(spec.get("char_filter", [])):
                if isinstance(f, dict):
                    analysis["char_filter"][f"{name}_c{i}"] = f
                    cl.append(f"{name}_c{i}")
                else:
                    cl.append(f)
            if cl:
                spec["char_filter"] = cl
        analysis["analyzer"][name] = spec
        props[f"f_{name}"] = {"type": "text", "analyzer": name, "fielddata": True, "term_vector": "with_positions_offsets"}
    return {"settings": {"number_of_shards": shards, "number_of_replicas": 0, "index.max_ngram_diff": 10, "index.max_shingle_diff": 10, "analysis": analysis},
            "mappings": {"properties": props}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9200")
    ap.add_argument("--stock", default="http://localhost:9201")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--index", default="fa")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    nodes = [Node(a.plugin), Node(a.stock)]
    body = settings(a.shards)
    if a.only:
        keep = set(a.only.split(","))
        body["mappings"]["properties"] = {k: v for k, v in body["mappings"]["properties"].items() if k[2:] in keep}
    for n in nodes:
        n.req("DELETE", f"/{a.index}")
        st, resp = n.req("PUT", f"/{a.index}", body)
        if st != 200:
            raise SystemExit(f"create failed on {n.url}: {json.dumps(resp)[:600]}")
    r = random.Random(a.seed)
    fields = list(body["mappings"]["properties"])
    lines = []
    for i in range(a.docs):
        t = text(r)
        lines.append(json.dumps({"index": {"_id": f"d{i}"}}))
        lines.append(json.dumps({f: t for f in fields}))
    payload = ("\n".join(lines) + "\n").encode()
    res = []
    for n in nodes:
        st, resp = n.req("POST", f"/{a.index}/_bulk?refresh=true", payload, raw=True)
        errs = [it["index"].get("error", {}).get("type") for it in resp.get("items", []) if "error" in it["index"]]
        res.append(errs)
    if res[0] != res[1]:
        print(f"bulk errors differ: plugin={res[0][:5]} stock={res[1][:5]}")
    for n in nodes:
        n.req("POST", f"/{a.index}/_forcemerge?max_num_segments=1")
        n.req("POST", f"/{a.index}/_refresh")
    bad = 0
    for f in fields:
        q = {"size": 0, "aggs": {"t": {"terms": {"field": f, "size": 100000, "order": [{"_key": "asc"}]}}}}
        rs = [n.req("POST", f"/{a.index}/_search", q) for n in nodes]
        if rs[0][0] != 200 or rs[1][0] != 200:
            if rs[0][0] != rs[1][0]:
                print(f"{f}: status plugin={rs[0][0]} stock={rs[1][0]}")
                bad += 1
            continue
        pt = {b["key"]: b["doc_count"] for b in rs[0][1]["aggregations"]["t"]["buckets"]}
        st_ = {b["key"]: b["doc_count"] for b in rs[1][1]["aggregations"]["t"]["buckets"]}
        if pt != st_:
            bad += 1
            only_p = sorted(set(pt) - set(st_))[:8]
            only_s = sorted(set(st_) - set(pt))[:8]
            cnt = [(k, pt[k], st_[k]) for k in pt if k in st_ and pt[k] != st_[k]][:5]
            print(f"{f}: TERMS differ plugin={len(pt)} stock={len(st_)} only_plugin={only_p} only_stock={only_s} count_diff={cnt}")
        # term vectors of a few documents: positions and offsets
        for i in r.sample(range(a.docs), min(5, a.docs)):
            tv = [n.req("GET", f"/{a.index}/_termvectors/d{i}?fields={f}&offsets=true&positions=true&term_statistics=false&field_statistics=false") for n in nodes]
            if tv[0][0] != 200 or tv[1][0] != 200:
                continue
            d = diff(tv[0][1].get("term_vectors"), tv[1][1].get("term_vectors"), ignore={"took"})
            if d:
                bad += 1
                print(f"{f}: TERMVECTORS d{i} differ: {d}")
                break
    print(f"analyzers checked: {len(fields)}, mismatching fields: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
