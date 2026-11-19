"""loop.py BASE SHAPE_SUBSTR SECS [CLIENTS]: closed loop of one shape (for profiling); prints qps."""
import os, sys, time, threading, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, "../../opensearch-plugin/e2e")); sys.path.insert(0, os.path.join(here, "../agg-sweep"))
import bench_nodes as bn, split
base, pat, secs = sys.argv[1], sys.argv[2], float(sys.argv[3]); cl = int(sys.argv[4]) if len(sys.argv) > 4 else 1
bn.BASE = base
sh = [s for s in split.all_shapes() if pat == s[0] + "/" + s[1] or pat in s[1]][0]
path = "/%s/_search?request_cache=false%s" % (sh[2], sh[4]); body = sh[3]
cnt = [0]; stop = time.time() + secs
def w():
    n = 0
    while time.time() < stop:
        bn.req("POST", path, body); n += 1
    cnt[0] += n
ts = [threading.Thread(target=w) for _ in range(cl)]
t = time.time(); [x.start() for x in ts]; [x.join() for x in ts]
print(json.dumps({"shape": sh[1], "qps": cnt[0] / (time.time() - t)}))
