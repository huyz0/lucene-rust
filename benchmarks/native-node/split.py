"""Where one search request's time goes, per shape: client round trip, node CPU per request, and (plugin) the
native query-phase time, native fetch time and whether the request ran natively or fell back.

  split.py BASE LABEL OUT [N]      serial, one keep-alive connection per shape; N timed requests (default 200)

Shapes: the 63 core shapes (ab.py, index `bench`) plus the feature families (families.py), and the floor
shapes `floor match_none` / `floor size0 count`. SHAPES=a,b selects by substring of "family/name".
Java request path per request = round-trip p50 minus native query and fetch time (plugin only); node CPU per
request comes from /proc/<node>/stat over the timed batch. The stats endpoint is read before and after each
batch (two extra requests, not timed)."""
import json
import os
import subprocess
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "../../opensearch-plugin/e2e"))
sys.path.insert(0, os.path.join(here, "../agg-sweep"))
import bench_nodes as bn  # noqa: E402
import e2e_sweep  # noqa: E402
import families  # noqa: E402

N = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("N", "200"))
WARM_S = float(os.environ.get("WARMUP_S", "1.5"))


def node_pid():
    out = subprocess.run(["pgrep", "-f", "org.opensearch.bootstrap.OpenSearch"], capture_output=True, text=True).stdout.split()
    return int(out[0])


def cpu_s(pid):
    f = open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def majflt(pid):
    return int(open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()[9])


def read_kb(pid):
    for line in open("/proc/%d/io" % pid):
        if line.startswith("read_bytes"):
            return int(line.split()[1]) / 1024
    return 0


def pstats():
    try:
        s = bn.req("GET", "/_plugins/accelerated/stats")
    except Exception:  # noqa: BLE001
        return None
    q = s.get("query_phase_nanos", {})
    f = s.get("fetch_nanos", {})
    return {"nat_ns": q.get("accelerated", 0), "nat_n": q.get("accelerated_count", 0),
            "fb_ns": q.get("fallback", 0), "fb_n": q.get("fallback_count", 0),
            "fetch_ns": f.get("accelerated", 0), "fetch_fb_ns": f.get("fallback", 0),
            "fetch_n": f.get("accelerated_count", 0), "fetch_fb_n": f.get("fallback_count", 0),
            "fallbacks": s.get("fallbacks", {})}


def all_shapes():
    out = []
    for name, body in bn.shapes() + e2e_sweep.shapes():
        out.append(("core", name, "bench", body, ""))
    out.extend(families.sh())
    out.append(("floor", "floor match_none", "bench", {"size": 0, "query": {"match_none": {}}}, ""))
    out.append(("floor", "floor size0 term count", "bench", {"size": 0, "track_total_hits": False, "query": {"term": {"tag": "c1"}}}, ""))
    want = [w for w in os.environ.get("SHAPES", "").split(",") if w]
    return [s for s in out if not want or any(w in s[0] + "/" + s[1] for w in want)]


def main():
    base, label, out = sys.argv[1:4]
    bn.BASE = base
    pid = node_pid()
    res = {}
    for fam, name, idx, body, params in all_shapes():
        path = "/%s/_search?request_cache=false%s" % (idx, params)
        last = []

        def one():
            last[:] = [bn.req("POST", path, body)]

        try:
            t = time.perf_counter()
            one()
            first = (time.perf_counter() - t) * 1000
            n = max(8, min(N, int(3000 / max(first, 0.1))))
            bn.bh.warm(one, min(20, n), WARM_S if first < 50 else 0)
            a, c0, io0, mf0 = pstats(), cpu_s(pid), read_kb(pid), majflt(pid)
            wall = bn.bh.measure(one, n, 0, 0)
            c1, b, io1, mf1 = cpu_s(pid), pstats(), read_kb(pid), majflt(pid)
        except Exception as e:  # noqa: BLE001
            res[fam + "/" + name] = {"error": str(e)[:200]}
            print("ERR", name, str(e)[:100], flush=True)
            continue
        r = {**bn.bh.summarize(wall), "cpu_ms": (c1 - c0) * 1000 / n, "disk_read_kb": (io1 - io0) / n, "major_faults": (mf1 - mf0) / n, "took": last[0].get("took")}
        if a and b:
            r["native_ms"] = (b["nat_ns"] - a["nat_ns"]) / 1e6 / n
            r["fallback_ms"] = (b["fb_ns"] - a["fb_ns"]) / 1e6 / n
            r["native_n"] = (b["nat_n"] - a["nat_n"]) / n
            r["fallback_n"] = (b["fb_n"] - a["fb_n"]) / n
            r["fetch_ms"] = (b["fetch_ns"] - a["fetch_ns"] + b["fetch_fb_ns"] - a["fetch_fb_ns"]) / 1e6 / n
            r["fallback_reasons"] = {k: v - a["fallbacks"].get(k, 0) for k, v in b["fallbacks"].items() if v != a["fallbacks"].get(k, 0)}
        res[fam + "/" + name] = r
        print("%-10s %-44s p50 %7.2f cpu %7.2f nat %s" % (fam, name[:44], r["median_ms"], r["cpu_ms"],
              "%.2f fb %.2f" % (r["native_ms"], r["fallback_ms"]) if "native_ms" in r else "-"), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
