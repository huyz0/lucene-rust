"""Concurrent closed-loop driver over the taxis shapes (taxis.py): qps and node CPU per request.

  python3 conc_taxis.py BASE SHAPE_SUBSTRING SECONDS NODE_PID [PROCS=2] [CONNS=4] [WARM=3]

Same load generator as conc.py (raw keep-alive sockets), one shape per call.
"""
import json
import multiprocessing as mp
import os
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taxis  # noqa: E402
from conc import cpu, proc  # noqa: E402

if __name__ == "__main__":
    base, sub, secs, pid = sys.argv[1:5]
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    conns = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    warm = float(sys.argv[7]) if len(sys.argv) > 7 else 3
    u = urlparse(base)
    name, body = next((n, b) for n, b in taxis.shapes() if n == sub)
    b = json.dumps(body).encode()
    req = (f"POST /bench/_search?request_cache=false HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(b)}\r\n\r\n").encode() + b
    qu = mp.Queue()
    t0 = time.perf_counter() + 1
    warm_at, stop_at = t0 + warm, t0 + warm + float(secs)
    ps = [mp.Process(target=proc, args=((u.hostname, u.port, [req], t0, stop_at, conns, warm_at, qu),)) for _ in range(procs)]
    for p in ps:
        p.start()
    time.sleep(max(0, warm_at - time.perf_counter()))
    c0, tt = cpu(pid), time.perf_counter()
    for p in ps:
        p.join()
    c1, dt = cpu(pid), time.perf_counter() - tt
    n = sum(qu.get() for _ in ps)
    print(json.dumps({"shape": name, "qps": round(n / dt), "cpu_ms_per_req": round((c1 - c0) / n * 1000, 3)}))
