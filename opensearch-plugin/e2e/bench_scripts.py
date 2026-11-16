"""Times the script, expression, function_score and java.time shapes of scripts_equal.py on a larger
index, for stock against the plugin (alternate the two nodes with benchmarks/native-node/native-node.sh).

  bench_scripts.py load BASE [DOCS]       index DOCS (default 400000) documents, force-merge to one segment
  bench_scripts.py time BASE OUT [ROUNDS] median and p90 per shape, written to OUT as JSON
  bench_scripts.py report OUT_STOCK OUT_PLUGIN...   the table
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchhttp as bh
import scripts_equal as se

PICK = [
    "stored script_score", "stored sort", "script_score min_score", "fs min_score fvf",
    "fvf l", "random l", "random _id", "random _id sum", "decay gauss d", "decay exp date", "decay linear geo",
    "expr score long", "expr score multi", "expr sort long", "expr score fns", "expr script query", "expr agg sum",
    "expr sort _score q term", "painless sort _score q term",
    "time score year", "time filter getters", "time sort plus", "time agg year", "time score truncate",
    "hammingbit long",
]


def load(base, n):
    se.N = n
    se.load(base)
    bh.req(base, "POST", "/sc/_forcemerge?max_num_segments=1")
    bh.req(base, "POST", "/sc/_refresh")


def timing(base, out, rounds):
    shapes = se.shapes()
    res = {}
    only = [x for x in os.environ.get("SHAPES", "").split("|") if x]
    for name in PICK:
        if only and not any(o in name for o in only):
            continue
        body = dict(shapes[name], size=10)
        path = "/sc/_search?request_cache=false&rest_total_hits_as_int=true"
        res[name] = bh.summarize(bh.measure(lambda: bh.req(base, "POST", path, body), rounds=rounds, warmup=20, warmup_s=1.0))
    json.dump(res, open(out, "w"))


def report(paths):
    sides = [[json.load(open(p)) for p in sorted(__import__("glob").glob(g))] for g in paths]
    names = [k for k in sides[0][0] if all(k in r for s in sides for r in s)]
    print("%-28s %10s" % ("shape (best median ms)", "stock") + "".join(" %10s %6s" % ("plugin" + str(i), "x") for i in range(len(sides) - 1)))
    for k in names:
        best = [min(r[k]["median_ms"] for r in s) for s in sides]
        print("%-28s %10.3f" % (k, best[0]) + "".join(" %10.3f %5.1fx" % (b, best[0] / b) for b in best[1:]))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "load":
        load(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400000)
    elif c == "time":
        timing(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 300)
    else:
        report(sys.argv[2:])
