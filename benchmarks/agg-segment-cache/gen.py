"""Loghub lines (bgl, hdfs, openssh, thunderbird; first 250 MB each) as log events, interleaved
as the engine study's Bench.read does, into bulk NDJSON files of 10k events.
  python3 gen.py DATA_DIR OUT_DIR
Each event: _id, @timestamp (7 days, evenly), source, host, level, bytes (line length), message.
"""
import json, os, random, re, sys

SETS = ["bgl", "hdfs", "openssh", "thunderbird"]
T0, SPAN = 1730000000000, 7 * 24 * 3600 * 1000
data, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
LEVEL = re.compile(r"\b(FATAL|SEVERE|ERROR|WARN(?:ING)?|INFO|DEBUG|FAILURE|failed|error)\b")

def host(src, line):
    t = line.split(" ", 6)
    if src == "thunderbird":
        return t[3] if len(t) > 3 else src
    if src == "bgl":
        return t[3][:6] if len(t) > 3 else src
    if src == "hdfs":
        return t[4] if len(t) > 4 else src
    return t[3] if len(t) > 3 else src

def level(line):
    m = LEVEL.search(line)
    if not m:
        return "info"
    w = m.group(1).lower()
    return {"warning": "warn", "severe": "error", "failure": "error", "failed": "error"}.get(w, w)

def lines():
    fs = [open(os.path.join(data, s + ".txt"), encoding="iso-8859-1") for s in SETS]
    done = [False] * len(SETS)
    live = len(SETS)
    rnd = random.Random(7)
    while live:
        s = rnd.randrange(len(SETS))
        if done[s]:
            continue
        line = fs[s].readline()
        if not line:
            done[s] = True
            live -= 1
            continue
        line = line.rstrip("\n")
        if line:
            yield SETS[s], line

events = list(lines())
n = len(events)
# The first LOAD events are the initial index; the rest arrive during the live phase.
LOAD = n - 200_000
print("events", n, "load", LOAD, flush=True)
step = SPAN / LOAD
batch, part = [], 0
def flush():
    global batch, part
    if batch:
        with open(os.path.join(out, f"bulk-{part:05d}.ndjson"), "w") as fh:
            fh.write("".join(batch))
        part += 1
        batch = []
for i, (src, line) in enumerate(events):
    if i == LOAD:
        flush()
        out = os.path.join(sys.argv[2], "live")
        os.makedirs(out, exist_ok=True)
        part = 0
    doc = {"@timestamp": int(T0 + i * step), "source": src, "host": host(src, line), "level": level(line),
           "bytes": len(line), "message": line}
    batch.append(json.dumps({"index": {"_id": str(i)}}) + "\n" + json.dumps(doc) + "\n")
    if len(batch) == 10_000 if i < LOAD else len(batch) == 5_000:
        flush()
flush()
json.dump({"events": n, "load": LOAD, "t0": T0, "step": step, "end": int(T0 + (LOAD - 1) * step)},
          open(os.path.join(sys.argv[2], "meta.json"), "w"))
