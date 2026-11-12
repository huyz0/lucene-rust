"""Interleave the four Loghub sets (first 250 MB each) into bulk NDJSON files of 5k docs."""
import json, os, random, sys
SRC = sys.argv[1]; OUT = sys.argv[2]
SETS = ["bgl", "hdfs", "openssh", "thunderbird"]
T0, SPAN = 1730000000000, 7 * 24 * 3600 * 1000
files = [open(os.path.join(SRC, s + ".txt"), encoding="latin-1") for s in SETS]
def host(src, line):
    t = line.split(" ", 6)
    if src == "bgl": return t[3][:6] if len(t) > 3 else src
    if src == "hdfs": return t[4] if len(t) > 4 else src
    return t[3] if len(t) > 3 else src
lines = []
done = [False] * 4; rnd = random.Random(7)
while not all(done):
    s = rnd.randrange(4)
    if done[s]: continue
    l = files[s].readline()
    if not l: done[s] = True; continue
    l = l.rstrip("\n")
    if not l: continue
    lines.append((SETS[s], l))
# The last line of a file cut at 250 MB may be partial.
n = len(lines); os.makedirs(OUT, exist_ok=True)
B = 5000
for b in range(0, n, B):
    with open(os.path.join(OUT, f"{b // B:05d}.ndjson"), "w") as f:
        for i in range(b, min(b + B, n)):
            src, l = lines[i]
            f.write('{"create":{}}\n')
            f.write(json.dumps({"@timestamp": T0 + i * SPAN // n, "line": i, "source": src, "host": host(src, l), "message": l}) + "\n")
print(n)
