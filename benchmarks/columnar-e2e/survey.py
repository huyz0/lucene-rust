"""Stage 4 field survey: for numeric/date fields of a corpus sample, how many docs have the field
(presence) and how many have more than one value (multi-valued), and whether the dataset's
own aggregation workload aggregates it.

  survey.py NAME INDEX_JSON OPS_JSON SAMPLE_BZ2_OR_NDJSON [MAX_DOCS]

INDEX_JSON: the workload's index.json (mapping types); OPS_JSON: its operations/default.json
(fields used inside "aggs"). The sample is read as a stream (a truncated bz2 is fine).
Fields found in the data but not in the mapping are dynamic: reported with type `dynamic`."""
import bz2
import collections
import json
import re
import sys

name, idx, ops, sample = sys.argv[1:5]
maxdocs = int(sys.argv[5]) if len(sys.argv) > 5 else 500_000


def mapped(props, prefix=""):
    out = {}
    for k, v in props.items():
        p = prefix + k
        if "properties" in v:
            out.update(mapped(v["properties"], p + "."))
        else:
            out[p] = v.get("type", "object")
            for sub, sv in v.get("fields", {}).items():
                out[p + "." + sub] = sv.get("type", "?")
    return out


def find_props(d):
    if isinstance(d, dict):
        if "properties" in d and isinstance(d["properties"], dict):
            return d["properties"]
        for v in d.values():
            r = find_props(v)
            if r:
                return r
    return None


idxtxt = re.sub(r"\{[%#].*?[%#]\}|\{\{.*?\}\}", "0", open(idx).read(), flags=re.S)
try:
    types = mapped(find_props(json.loads(idxtxt)) or {})
except Exception as e:
    types = {}
    print("mapping parse failed:", e, file=sys.stderr)

optxt = open(ops).read()
agg_fields = set()
for m in re.finditer(r'"(?:aggs|aggregations)"', optxt):
    seg = optxt[m.start(): m.start() + 3000]
    agg_fields.update(re.findall(r'"field"\s*:\s*"([^"]+)"', seg))

NUM = {"long*", "double*", "date*", "boolean*", "long", "integer", "short", "byte", "double", "float", "half_float", "scaled_float", "date", "unsigned_long", "ip", "boolean"}
present, multi, n = collections.Counter(), collections.Counter(), 0
seen = collections.defaultdict(collections.Counter)
DATE = re.compile(r"^\d{4}-\d\d-\d\d[T ]\d\d:\d\d")


def kind(x):
    if isinstance(x, bool):
        return "boolean"
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "date" if DATE.match(x) else "string"
    return "other"



def walk(o, prefix):
    for k, v in o.items():
        p = prefix + k
        if isinstance(v, dict):
            walk(v, p + ".")
        elif isinstance(v, list):
            vals = [x for x in v if x is not None]
            if not vals:
                continue
            present[p] += 1
            if len(vals) > 1:
                multi[p] += 1
            seen[p][kind(vals[0])] += 1
        elif v is not None:
            present[p] += 1
            seen[p][kind(v)] += 1


op = bz2.open if sample.endswith(".bz2") else open
with op(sample, "rt", encoding="utf-8", errors="replace") as fh:
    try:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or "index" in d and len(d) == 1:
                continue
            n += 1
            walk(d, "")
            if n >= maxdocs:
                break
    except (OSError, EOFError):
        pass

rows = []
for f in present:
    t = types.get(f) or types.get(f.split(".")[0])
    if not t:  # no mapping parsed: what the sampled values look like
        t = "inferred " + seen[f].most_common(1)[0][0] if seen[f] else "dynamic"
    if t.startswith("inferred") and t.split()[1] in ("int", "float", "date", "boolean"):
        t = {"int": "long", "float": "double"}.get(t.split()[1], t.split()[1]) + "*"
    rows.append((f, t, present[f] / n, multi[f] / n, f in agg_fields or any(a.startswith(f + ".") for a in agg_fields)))
rows.sort(key=lambda r: (not r[4], r[1] not in NUM, r[0]))
print(f"## {name}: {n} sampled docs; aggregated by the dataset's own workload: {sorted(agg_fields)}")
print("| field | type | present | multi-valued | aggregated |")
print("|---|---|---|---|---|")
for f, t, p, m, a in rows:
    if t in NUM or a:
        print(f"| {f} | {t} | {p*100:.1f}% | {m*100:.1f}% | {'yes' if a else ''} |")
