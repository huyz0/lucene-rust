"""Concurrent closed-loop driver: qps and node CPU per request for one text_bench shape.

  python3 conc.py BASE OUT_DIR SHAPE SECONDS NODE_PID [PROCS=2] [CONNS=4] [WARM=3]

PROCS processes of CONNS keep-alive connections each (asyncio, raw sockets: the client's own cost per
request stays far under the node's). Prints one JSON line: qps, cpu_ms_per_req (node utime+stime
over the timed window, divided by requests), and the shape.
"""
import asyncio
import json
import multiprocessing as mp
import os
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_bench import body_for  # noqa: E402


async def worker(host, port, reqs, stop_at, count):
    r, w = await asyncio.open_connection(host, port)
    i = 0
    while time.perf_counter() < stop_at:
        body = reqs[i % len(reqs)]
        i += 1
        w.write(body)
        await w.drain()
        head = await r.readuntil(b"\r\n\r\n")
        n = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                n = int(line.split(b":")[1])
        await r.readexactly(n)
        count[0] += 1
    w.close()


def proc(args):
    host, port, reqs, start_at, stop_at, conns, warm_at, q = args
    count = [0]
    warm = [0]

    async def main():
        ts = [asyncio.create_task(worker(host, port, reqs, stop_at, count)) for _ in range(conns)]
        await asyncio.sleep(max(0, warm_at - time.perf_counter()))
        warm[0] = count[0]
        await asyncio.gather(*ts)

    asyncio.run(main())
    q.put(count[0] - warm[0])


def cpu(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


if __name__ == "__main__":
    base, out, shape, secs, pid = sys.argv[1:6]
    procs = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    conns = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    warm = float(sys.argv[8]) if len(sys.argv) > 8 else 3
    u = urlparse(base)
    Q = json.load(open(os.path.join(out, "queries.json")))
    reqs = []
    for q in Q[shape]:
        b = json.dumps(body_for(shape, q)).encode()
        reqs.append(
            (f"POST /txt/_search?request_cache=false HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
             f"Content-Length: {len(b)}\r\n\r\n").encode() + b)
    qu = mp.Queue()
    t0 = time.perf_counter() + 1
    warm_at, stop_at = t0 + warm, t0 + warm + float(secs)
    ps = [mp.Process(target=proc, args=((u.hostname, u.port, reqs[k::procs] or reqs, t0, stop_at, conns, warm_at, qu),))
          for k in range(procs)]
    for p in ps:
        p.start()
    time.sleep(max(0, warm_at - time.perf_counter()))
    c0, tt = cpu(pid), time.perf_counter()
    for p in ps:
        p.join()
    c1, dt = cpu(pid), time.perf_counter() - tt
    n = sum(qu.get() for _ in ps)
    print(json.dumps({"shape": shape, "qps": round(n / dt), "cpu_ms_per_req": round((c1 - c0) / n * 1000, 3)}))
