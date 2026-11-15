"""Shared HTTP client and timing helpers for every benchmark that talks to an OpenSearch node.

A search on the fast paths takes 0.3-3 ms. urllib opens a TCP connection per request (about 3.5 ms
of client time on the cloud box), http.client keep-alive still costs about 1 ms, and both sides pay
it equally, so a client like that hides the difference between the engines. This client keeps one
raw HTTP/1.1 socket per (base URL, thread), sets TCP_NODELAY, and does the bare minimum per request.

  Client(base).json(method, path, body)       parsed JSON (RuntimeError on status >= 300)
  req(base, method, path, body, ndjson=...)   same, on this thread's pooled connection
  warm(fn, min_n, min_s)                      call fn until both min_n calls and min_s seconds passed
  measure(fn, rounds, warmup, warmup_s)       warm, then time `rounds` calls: list of milliseconds
  summarize(ms)                               n, min, p50, p90, p99, mean, max

Timing rules the callers follow: warm until the JIT has settled (a node that was just started needs
seconds, not 30 requests), time many more samples than a few dozen for anything under 3 ms, report
p50 and p90/p99 from the sorted samples, and drive concurrency with a separate driver
(benchmarks/native-node/conc.py) so the client's CPU is measured and kept off the node's cores.
"""
import json
import re
import socket
import statistics
import threading
import time

_LOCAL = threading.local()
_CL = re.compile(rb"(?im)^content-length:\s*(\d+)")
_CHUNKED = re.compile(rb"(?im)^transfer-encoding:\s*chunked")
_CLOSE = re.compile(rb"(?im)^connection:\s*close")


def _split(base):
    hp = base.split("//", 1)[-1].rstrip("/")
    host, _, port = hp.partition(":")
    return ("127.0.0.1" if host == "localhost" else host), int(port or 80)


class Client:
    """One persistent HTTP/1.1 connection, reopened when the server has closed it. Not thread-safe:
    use `for_thread(base)` or `req` from threaded code."""

    def __init__(self, base, timeout=600):
        self.addr = _split(base)
        self.timeout = timeout
        self.s = None
        self.buf = b""

    def _open(self):
        s = socket.create_connection(self.addr, timeout=self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s, self.buf = s, b""

    def close(self):
        if self.s is not None:
            try:
                self.s.close()
            finally:
                self.s = None

    def _read_until(self, marker):
        while marker not in self.buf:
            c = self.s.recv(1 << 20)
            if not c:
                raise ConnectionError("closed")
            self.buf += c

    def _read_exact(self, n):
        while len(self.buf) < n:
            c = self.s.recv(1 << 20)
            if not c:
                raise ConnectionError("closed")
            self.buf += c
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _read_response(self):
        self._read_until(b"\r\n\r\n")
        head, _, self.buf = self.buf.partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        if _CHUNKED.search(head):
            parts = []
            while True:
                self._read_until(b"\r\n")
                line, _, self.buf = self.buf.partition(b"\r\n")
                n = int(line.split(b";")[0], 16)
                if n == 0:
                    self._read_until(b"\r\n")
                    self.buf = self.buf.partition(b"\r\n")[2]
                    break
                parts.append(self._read_exact(n))
                self._read_exact(2)
            data = b"".join(parts)
        else:
            m = _CL.search(head)
            data = self._read_exact(int(m.group(1))) if m else b""
        if _CLOSE.search(head):
            self.close()
        return status, data

    def request(self, method, path, body=None, ndjson=False, timeout=None):
        """(status, body bytes). `body`: bytes, str, or a JSON-able object."""
        if body is None:
            data = b""
        elif isinstance(body, bytes):
            data = body
        elif isinstance(body, str):
            data = body.encode()
        else:
            data = json.dumps(body).encode()
        ctype = "application/x-ndjson" if ndjson else "application/json"
        head = f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: {ctype}\r\nContent-Length: {len(data)}\r\n\r\n".encode()
        for attempt in (0, 1):
            reused = self.s is not None
            if not reused:
                self._open()
            self.s.settimeout(timeout or self.timeout)
            try:
                self.s.sendall(head + data)
                return self._read_response()
            except (ConnectionError, BrokenPipeError):
                # An idle keep-alive connection the server dropped: retry once on a fresh one. A request
                # that failed on a fresh connection is a real failure.
                self.close()
                if not reused or attempt:
                    raise
            except Exception:
                self.close()
                raise

    def json(self, method, path, body=None, ndjson=False, timeout=None):
        status, data = self.request(method, path, body, ndjson, timeout)
        if status >= 300:
            raise RuntimeError(f"{method} {path}: {status} {data[:500].decode(errors='replace')}")
        return json.loads(data) if data else None


def for_thread(base):
    """This thread's connection to `base`."""
    pool = getattr(_LOCAL, "pool", None)
    if pool is None:
        pool = _LOCAL.pool = {}
    c = pool.get(base)
    if c is None:
        c = pool[base] = Client(base)
    return c


def req(base, method, path, body=None, ndjson=False, timeout=600):
    return for_thread(base).json(method, path, body, ndjson, timeout)


def warm(fn, min_n=30, min_s=2.0):
    """Call fn until at least min_n calls and min_s seconds have passed: a cold JVM settles by time spent
    in the code, and a fast shape gets thousands of calls in two seconds, a slow one few."""
    t0 = time.perf_counter()
    n = 0
    while n < min_n or time.perf_counter() - t0 < min_s:
        fn()
        n += 1
    return n


def measure(fn, rounds=100, warmup=30, warmup_s=2.0):
    """Warm up, then time `rounds` calls of fn(): milliseconds, in call order."""
    warm(fn, warmup, warmup_s)
    out = []
    for _ in range(rounds):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def pct(sorted_ms, p):
    return sorted_ms[min(len(sorted_ms) - 1, max(0, int(round(p * (len(sorted_ms) - 1)))))]


def summarize(ms):
    s = sorted(ms)
    return {"n": len(s), "min_ms": round(s[0], 3), "median_ms": round(statistics.median(s), 3),
            "p90_ms": round(pct(s, 0.9), 3), "p99_ms": round(pct(s, 0.99), 3),
            "mean_ms": round(statistics.mean(s), 3), "max_ms": round(s[-1], 3)}
