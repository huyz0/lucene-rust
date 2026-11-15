#!/usr/bin/env python3
"""Painless differential: random typed expressions and small programs, evaluated per document as
`script_fields` (and as a script_score / sort / terms-aggregation key) on a plugin node and a
stock node; values (including float formatting) and errors must be identical.

    python3 opensearch-plugin/e2e/fuzz_diff.py --queries 0 --docs 300   # builds index fz on both
    python3 opensearch-plugin/e2e/fuzz_painless.py --seed 1 --cases 400
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node, close  # noqa: E402

LONGS = ["n", "i", "sh"]
DOUBLES = ["price", "f", "sf", "hf"]


class G:
    def __init__(self, r, simple=False):
        self.r = r
        self.simple = simple
        self.locals = {"long": [], "double": [], "String": [], "boolean": [], "int": []}

    def long_(self, d=0):
        r = self.r
        if d > 3 or r.random() < .25:
            f_ = r.choice(LONGS)
            return r.choice([f"(doc['{f_}'].size() == 0 ? {r.choice(['0L', '-1L', '7L'])} : doc['{f_}'].value)",
                             r.choice(["0L", "1L", "-1L", "42L", "2147483647L", "-2147483648L", "9223372036854775807L", "1000000007L", "255L"])] + self.locals["long"])
        k = r.choice(["add", "sub", "mul", "div", "mod", "shl", "shr", "ushr", "and", "or", "xor", "neg", "not", "ternary", "abs", "max", "min", "cast_d", "len", "parse", "round", "floorDiv", "pow", "idx", "hash", "compare", "mulexact", "bitcount", "ch"])
        a, b = self.long_(d + 1), self.long_(d + 1)
        if k == "add": return f"({a} + {b})"
        if k == "sub": return f"({a} - {b})"
        if k == "mul": return f"({a} * {b})"
        if k == "div": return f"({a} / ({b} == 0L ? 1L : {b}))" if r.random() < .8 else f"({a} / {b})"
        if k == "mod": return f"({a} % ({b} == 0L ? 3L : {b}))" if r.random() < .8 else f"({a} % {b})"
        if k == "shl": return f"({a} << {r.randint(0, 70)})"
        if k == "shr": return f"({a} >> {r.randint(0, 70)})"
        if k == "ushr": return f"({a} >>> {r.randint(0, 70)})"
        if k == "and": return f"({a} & {b})"
        if k == "or": return f"({a} | {b})"
        if k == "xor": return f"({a} ^ {b})"
        if k == "neg": return f"(-{a})"
        if k == "not": return f"(~{a})"
        if k == "ternary": return f"({self.bool_(d + 1)} ? {a} : {b})"
        if k == "abs": return f"Math.abs({a})"
        if k == "max": return f"Math.max({a}, {b})"
        if k == "min": return f"Math.min({a}, {b})"
        if k == "cast_d": return f"(long) {self.double_(d + 1)}"
        if k == "len": return f"(long) {self.str_(d + 1)}.length()"
        if k == "parse": return f"(long) {self.int_(d + 1)}"
        if k == "round": return f"Math.round({self.double_(d + 1)})"
        if k == "floorDiv": return f"Math.floorDiv({a}, ({b} == 0L ? 1L : {b}))"
        if k == "pow": return f"(long) Math.pow({self.double_(d + 1)}, {r.randint(0, 4)})"
        if k == "idx": return f"(long) {self.str_(d + 1)}.indexOf({r.choice(['\"a\"', '\"e\"', '\"\"', '\"zz\"'])})"
        if k == "hash": return f"(long) {self.str_(d + 1)}.hashCode()"
        if k == "compare": return f"(long) {self.str_(d + 1)}.compareTo({self.str_(d + 1)})"
        if k == "mulexact": return f"Math.multiplyExact({a}, {b})"
        if k == "bitcount": return f"(long) Long.bitCount({a})"
        if k == "ch": return f"(long) {self.str_(d + 1)}.charAt(0)"

    def int_(self, d=0):
        r = self.r
        if d > 3 or r.random() < .4:
            return r.choice(["0", "1", "-1", "100", "2147483647", "-2147483648", "65535", "7"] + self.locals["int"])
        k = r.choice(["add", "mul", "sub", "div", "cast", "parse", "len", "shift", "mod"])
        a, b = self.int_(d + 1), self.int_(d + 1)
        if k == "add": return f"({a} + {b})"
        if k == "mul": return f"({a} * {b})"
        if k == "sub": return f"({a} - {b})"
        if k == "div": return f"({a} / ({b} == 0 ? 1 : {b}))"
        if k == "mod": return f"({a} % ({b} == 0 ? 5 : {b}))"
        if k == "cast": return f"(int) {self.long_(d + 1)}"
        if k == "parse": return f"Integer.parseInt({r.choice(['\"12\"', '\"-7\"', '\"0\"', '\"x1\"', '\"2147483648\"', '\"\"', '\" 5\"'])})"
        if k == "len": return f"{self.str_(d + 1)}.length()"
        if k == "shift": return f"({a} << {r.randint(0, 40)})"

    def double_(self, d=0):
        r = self.r
        if d > 3 or r.random() < .25:
            f_ = r.choice(DOUBLES)
            return r.choice([f"(doc['{f_}'].size() == 0 ? {r.choice(['0.0', '-1.5', '2.0'])} : doc['{f_}'].value)",
                             r.choice(["0.0", "-0.0", "1.5", "-2.5", "0.1", "1e10", "1e-10", "3.141592653589793", "100.0", "0.5", "1e300", "4.9e-324", "123456789.123456789", "(double) Long.MAX_VALUE"])] + self.locals["double"])
        k = r.choice(["add", "sub", "mul", "div", "mod", "neg", "ternary", "abs", "sqrt", "log", "exp", "pow", "floor", "ceil", "rint", "sin", "cos", "tan", "atan2", "hypot", "cast", "signum", "max", "min", "log10", "cbrt", "tanh", "toRadians", "fromstr", "ulp", "nan", "cosh"])
        a, b = self.double_(d + 1), self.double_(d + 1)
        if k == "add": return f"({a} + {b})"
        if k == "sub": return f"({a} - {b})"
        if k == "mul": return f"({a} * {b})"
        if k == "div": return f"({a} / {b})"
        if k == "mod": return f"({a} % {b})"
        if k == "neg": return f"(-{a})"
        if k == "ternary": return f"({self.bool_(d + 1)} ? {a} : {b})"
        if k in ("abs", "sqrt", "log", "exp", "floor", "ceil", "rint", "sin", "cos", "tan", "signum", "log10", "cbrt", "tanh", "toRadians", "ulp", "cosh"):
            fn = {"toRadians": "toRadians"}.get(k, k)
            return f"Math.{fn}({a})"
        if k == "pow": return f"Math.pow({a}, {b})"
        if k == "atan2": return f"Math.atan2({a}, {b})"
        if k == "hypot": return f"Math.hypot({a}, {b})"
        if k == "cast": return f"(double) {self.long_(d + 1)}"
        if k == "max": return f"Math.max({a}, {b})"
        if k == "min": return f"Math.min({a}, {b})"
        if k == "fromstr": return f"Double.parseDouble({r.choice(['\"1.5\"', '\"-0\"', '\"1e3\"', '\"NaN\"', '\"Infinity\"', '\"abc\"', '\" 2 \"', '\"0x10\"', '\".5\"'])})"
        if k == "nan": return r.choice(["Double.NaN", "Double.POSITIVE_INFINITY", "Double.NEGATIVE_INFINITY", "Double.MIN_VALUE", "Double.MAX_VALUE"])

    def str_(self, d=0):
        r = self.r
        if d > 3 or r.random() < .3:
            return r.choice([f"(doc['tag'].size() == 0 ? 'none' : doc['tag'].value)", "'Hello World'", "''", "'a,b,,c'", "'  pad  '", "'ünï©ödé \U0001f600'", "'MiXeD'", "'a.b.c'", "'x'"] + self.locals["String"])
        k = r.choice(["concat", "concat_n", "concat_d", "upper", "lower", "sub", "replace", "trim", "ternary", "valueOf", "join", "repeat", "tostr_l", "format", "substr2", "reverse", "split", "regex", "hex", "date"])
        a, b = self.str_(d + 1), self.str_(d + 1)
        if k == "concat": return f"({a} + {b})"
        if k == "concat_n": return f"({a} + {self.long_(d + 1)})"
        if k == "concat_d": return f"({a} + {self.double_(d + 1)})"
        if k == "upper": return f"{a}.toUpperCase()"
        if k == "lower": return f"{a}.toLowerCase()"
        if k == "sub": return f"{a}.substring(Math.min({a}.length(), {r.randint(0, 3)}))"
        if k == "substr2": return f"{a}.substring(0, Math.min({a}.length(), {r.randint(0, 5)}))"
        if k == "replace": return f"{a}.replace({r.choice(['\"a\"', '\"l\"', '\".\"', '\"\"', '\"é\"'])}, {r.choice(['\"X\"', '\"\"', '\"yy\"'])})"
        if k == "trim": return f"{a}.trim()"
        if k == "ternary": return f"({self.bool_(d + 1)} ? {a} : {b})"
        if k == "valueOf": return r.choice([f"String.valueOf({self.double_(d + 1)})", f"String.valueOf({self.long_(d + 1)})", f"Double.toString({self.double_(d + 1)})", f"Long.toString({self.long_(d + 1)})", f"Float.toString((float) {self.double_(d + 1)})"])
        if k == "join": return f"String.join('-', {a}, {b})"
        if k == "repeat": return f"({a} + {a})"
        if k == "tostr_l": return f"Long.toString({self.long_(d + 1)}, {r.choice([2, 8, 16, 36])})"
        if k == "format": return r.choice([f"String.format('%d|%5.2f|%s', {self.long_(d + 1)}, {self.double_(d + 1)}, {a})", f"String.format('%08.3f', {self.double_(d + 1)})", f"String.format('%x', {self.long_(d + 1)})", f"String.format('%e', {self.double_(d + 1)})", f"String.format('%,d', {self.long_(d + 1)})"])
        if k == "reverse": return f"new StringBuilder({a}).reverse().toString()"
        if k == "split": return f"({a}.splitOnToken(',').length > 1 ? 'multi' : 'single')"
        if k == "regex": return r.choice([f"/[aeiou]/.matcher({a}).replaceAll('*')", f"/(\\w)(\\w)/.matcher({a}).replaceFirst('$2$1')", f"(({a} =~ /l+/) ? 'm' : 'n')"])
        if k == "hex": return f"Integer.toHexString({self.int_(d + 1)})"
        if k == "date": return f"(doc['ts'].size() == 0 ? 'nodate' : String.valueOf(doc['ts'].value.{r.choice(['getYear()', 'getMonthValue()', 'getDayOfMonth()', 'getHour()', 'getDayOfWeekEnum()', 'toString()', 'getDayOfYear()', 'getZone()', 'getMinute()'])}))"

    def bool_(self, d=0):
        r = self.r
        if d > 3 or r.random() < .25:
            return r.choice(["true", "false", "(doc['flag'].size() > 0 && doc['flag'].value)", "(doc['tag'].size() == 0)", "(doc['n'].size() > 0)"])
        k = r.choice(["lt", "gt", "eq", "and", "or", "not", "seq", "contains", "startsWith", "eqd", "isnan", "le", "ne", "regex", "empty", "ge"])
        if k == "lt": return f"({self.long_(d + 1)} < {self.long_(d + 1)})"
        if k == "gt": return f"({self.double_(d + 1)} > {self.double_(d + 1)})"
        if k == "le": return f"({self.double_(d + 1)} <= {self.long_(d + 1)})"
        if k == "ge": return f"({self.long_(d + 1)} >= {self.double_(d + 1)})"
        if k == "eq": return f"({self.long_(d + 1)} == {self.long_(d + 1)})"
        if k == "ne": return f"({self.int_(d + 1)} != {self.int_(d + 1)})"
        if k == "eqd": return f"({self.double_(d + 1)} == {self.double_(d + 1)})"
        if k == "and": return f"({self.bool_(d + 1)} && {self.bool_(d + 1)})"
        if k == "or": return f"({self.bool_(d + 1)} || {self.bool_(d + 1)})"
        if k == "not": return f"(!{self.bool_(d + 1)})"
        if k == "seq": return f"{self.str_(d + 1)}.equals({self.str_(d + 1)})"
        if k == "contains": return f"{self.str_(d + 1)}.contains({r.choice(['\"a\"', '\"o\"', '\"\"', '\"World\"'])})"
        if k == "startsWith": return f"{self.str_(d + 1)}.startsWith({r.choice(['\"H\"', '\"\"', '\"x\"', '\"é\"'])})"
        if k == "isnan": return f"Double.isNaN({self.double_(d + 1)})"
        if k == "regex": return f"({self.str_(d + 1)} ==~ /.*[a-m].*/)"
        if k == "empty": return f"{self.str_(d + 1)}.isEmpty()"

    def program(self):
        """Statements with locals, loops, lists, maps; ends with a typed return."""
        r = self.r
        if self.simple:
            return f"return {self.double_() if r.random() < .5 else self.long_()};"
        out = []
        for _ in range(r.randint(0, 3)):
            ty = r.choice(["long", "double", "String", "boolean", "int"])
            name = f"v{len(out)}"
            e = {"long": self.long_, "double": self.double_, "String": self.str_, "boolean": self.bool_, "int": self.int_}[ty]()
            out.append(f"{ty} {name} = {e};")
            self.locals[ty].append(name) if ty != "boolean" else None
        shape = r.choice(["expr", "expr", "loop", "list", "map", "while", "def", "switchlike", "trycatch", "stream", "foreach"])
        ret = r.choice(["long_", "double_", "str_", "bool_"])
        final = getattr(self, ret)()
        if shape == "loop":
            out.append(f"long acc = 0; for (int k = 0; k < {r.randint(0, 20)}; k++) {{ acc += (k * {self.long_(2)}) % 1000; }}")
            final = f"acc + {self.long_(2)}" if ret == "long_" else final
            ret = "long_" if ret == "long_" else ret
        elif shape == "list":
            out.append(f"List l = new ArrayList(); for (int k = 0; k < {r.randint(0, 6)}; k++) {{ l.add(k * {self.int_(3)}); }} def total = 0; for (def x : l) {{ total += x; }}")
            final = "total" if ret == "long_" else final
        elif shape == "map":
            out.append(f"Map m = new HashMap(); m.put('a', {self.long_(3)}); m.put('b', {self.double_(3)}); m.put('c', {self.str_(3)});")
            final = r.choice(["m.get('a')", "m.get('c')", "m.get('b')", "m.size()", "m.containsKey('z')", "m.getOrDefault('z', 'dflt')"])
        elif shape == "while":
            out.append(f"int w = {r.randint(1, 30)}; long acc = 1; while (w > 0) {{ acc = (acc * 31 + w) % 1000003L; w -= {r.randint(1, 4)}; }}")
            final = "acc"
        elif shape == "def":
            out.append(f"def d = {self.long_(2)}; def e = {self.double_(2)}; def s = {self.str_(2)}; def mix = d + e; def cat = s + d;")
            final = r.choice(["mix", "cat", "d", "e", "(d < e ? d : e)", "(d == e)", "(s.length() + d)"])
        elif shape == "switchlike":
            out.append(f"int sel = (int) (Math.abs({self.long_(2)}) % 4); String res; if (sel == 0) {{ res = 'zero'; }} else if (sel == 1) {{ res = 'one'; }} else if (sel == 2) {{ res = {self.str_(2)}; }} else {{ res = 'many'; }}")
            final = "res"
        elif shape == "trycatch":
            out.append(f"def val; try {{ val = {self.long_(2)} / {self.long_(2)}; }} catch (ArithmeticException ex) {{ val = -999; }} catch (Exception ex) {{ val = -998; }}")
            final = "val"
        elif shape == "stream":
            out.append(f"List l = [{self.long_(3)}, {self.long_(3)}, {self.long_(3)}]; long mx = l.stream().mapToLong(x -> x).max().getAsLong();")
            final = "mx"
        elif shape == "foreach":
            out.append("def acc = ''; for (def t : ['x', 'yy', doc['tag'].size() == 0 ? 'none' : doc['tag'].value]) { acc += t.length(); }")
            final = "acc"
        return " ".join(out) + f" return {final};"


def run(plugin, stock, index, source, kind):
    if kind == "fields":
        body = {"size": 40, "sort": [{"_seq_no": "asc"}], "_source": False, "script_fields": {"v": {"script": {"source": source, "lang": "painless"}}}}
    elif kind == "score":
        body = {"size": 40, "query": {"function_score": {"query": {"match_all": {}}, "script_score": {"script": {"source": "double __r = ((Number) (new Object() { Object x() { return 0; } }).x()).doubleValue(); return 1.0;"}}}}, "sort": [{"_seq_no": "asc"}]}
        body["query"]["function_score"]["script_score"]["script"]["source"] = source
    elif kind == "sort":
        body = {"size": 40, "sort": [{"_script": {"type": "number", "script": {"source": source}, "order": "asc"}}, {"_seq_no": "asc"}], "_source": False}
    elif kind == "agg":
        body = {"size": 0, "aggs": {"t": {"terms": {"script": {"source": source}, "size": 50, "order": [{"_key": "asc"}]}}}}
    elif kind == "aggsum":
        body = {"size": 0, "aggs": {"t": {"extended_stats": {"script": {"source": source}}}, "h": {"histogram": {"interval": 50, "min_doc_count": 1, "script": {"source": source}}}}}
    elif kind == "scorequery":
        body = {"size": 40, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}], "query": {"script_score": {"query": {"range": {"i": {"gte": -20}}}, "script": {"source": source}}}}
    else:
        body = {"size": 40, "sort": [{"_seq_no": "asc"}], "_source": False, "query": {"script": {"script": {"source": source}}}}
    p = plugin.req("POST", f"/{index}/_search", body)
    s = stock.req("POST", f"/{index}/_search", body)
    return body, p, s


def values(resp, kind):
    if kind == "fields":
        return [h.get("fields", {}).get("v") for h in resp["hits"]["hits"]]
    if kind == "aggsum":
        t = resp["aggregations"]["t"]
        return [t.get("count"), t.get("min"), t.get("max"), t.get("sum"), t.get("sum_of_squares"), [(b["key"], b["doc_count"]) for b in resp["aggregations"]["h"]["buckets"]]]
    if kind == "scorequery":
        return [(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]
    if kind in ("score", "sort", "filter"):
        return [(h["_id"], h.get("_score") if kind == "score" else h.get("sort") if kind == "sort" else None) for h in resp["hits"]["hits"]]
    return [(b["key"], b["doc_count"]) for b in resp["aggregations"]["t"]["buckets"]]


def same_val(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_val(x, y) for x, y in zip(a, b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same_val(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return close(a, b, 1e-9) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b
    return a == b


def err_sig(resp):
    e = resp[1].get("error", {})
    c = (e.get("root_cause") or [e])[0]
    return (c.get("type"), (c.get("reason") or "")[:100])


def native(node):
    st, r = node.req("GET", "/_plugins/lucene_rust/stats")
    return r.get("accelerated_queries") if st == 200 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9200")
    ap.add_argument("--stock", default="http://localhost:9201")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--simple", action="store_true", help="expression-only scripts (the native Painless subset)")
    ap.add_argument("--index", default="fz")
    ap.add_argument("--out", default="fuzz-painless.jsonl")
    a = ap.parse_args()
    plugin, stock = Node(a.plugin), Node(a.stock)
    r = random.Random(a.seed)
    out = open(a.out, "a")
    n_bad = n_err = n_native = 0
    nat, tot = {}, {}
    for ci in range(a.cases):
        g = G(r, a.simple)
        kind = r.choice(["fields", "score", "score", "sort", "sort", "agg", "filter", "filter", "filter", "aggsum", "aggsum", "scorequery", "scorequery"])
        src = g.program()
        if kind == "filter":
            src = src.replace("return ", "return (", 1) if False else src
            src = src.rsplit("return ", 1)[0] + f"return {g.bool_()};" if r.random() < .8 else src
        if kind in ("score", "scorequery"):
            src = src.rsplit("return ", 1)[0] + f"return Math.abs({g.double_()}) + 0.5;"
        if kind in ("sort", "aggsum"):
            src = src.rsplit("return ", 1)[0] + f"return {g.double_()};"
        st0 = native(plugin)
        body, p, s = run(plugin, stock, a.index, src, kind)
        st1 = native(plugin)
        if st0 is not None and st1 is not None:
            nat[kind] = nat.get(kind, 0) + (1 if st1 > st0 else 0)
            tot[kind] = tot.get(kind, 0) + 1
        if p[0] == -1 or s[0] == -1:
            print("node down", p[0], s[0])
            break
        msg = None
        if p[0] != s[0]:
            msg = f"status plugin={p[0]} stock={s[0]}: {json.dumps(p[1])[:200]} || {json.dumps(s[1])[:200]}"
        elif p[0] != 200:
            n_err += 1
            if err_sig(p)[0] != err_sig(s)[0]:
                msg = f"error type plugin={err_sig(p)} stock={err_sig(s)}"
        else:
            if kind in ("score", "scorequery"):
                pv, sv = [(i, x) for i, x in values(p[1], kind)], [(i, x) for i, x in values(s[1], kind)]
            else:
                pv, sv = values(p[1], kind), values(s[1], kind)
            if not same_val(pv, sv):
                msg = f"values differ: plugin={str(pv)[:300]} stock={str(sv)[:300]}"
        if msg:
            n_bad += 1
            rec = {"kind": kind, "source": src, "detail": msg, "seed": a.seed}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps(rec)[:1200], flush=True)
    print(json.dumps({"cases": a.cases, "findings": n_bad, "both_error": n_err, "native_of_total": {k: f"{nat.get(k, 0)}/{tot[k]}" for k in tot}}))


if __name__ == "__main__":
    sys.exit(main())
