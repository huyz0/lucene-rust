"""warmcurve.py BASE OUT_DIR SHAPE [BLOCKS=20] [BLOCK=50]: serial latency per block of requests right after start (the cold-JVM curve)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_bench import Conn, body_for
base, out, shape = sys.argv[1:4]
blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 20
bs = int(sys.argv[5]) if len(sys.argv) > 5 else 50
Q = json.load(open(os.path.join(out, "queries.json")))[shape]
c = Conn(base)
res = []
n = 0
for b in range(blocks):
    t = time.perf_counter()
    for _ in range(bs):
        c.call("POST", "/txt/_search?request_cache=false", body_for(shape, Q[n % len(Q)]))
        n += 1
    res.append(round((time.perf_counter() - t) / bs * 1000, 2))
print(shape, res)
