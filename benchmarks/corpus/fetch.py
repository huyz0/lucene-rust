#!/usr/bin/env python3
"""Corpora for the end-to-end benches, fetched or generated without Docker or a login.

  fetch.py synthetic DIR DOCS            seeded Zipf text + numerics + geo bulk bodies (no download)
  fetch.py loghub DIR [--mb 250]         Loghub bgl, hdfs, openssh, thunderbird lines (Hugging Face)
  fetch.py bgl DIR                       Loghub BGL lines, the whole file
  fetch.py taxi DIR                      NYC yellow taxi 2025-01 and 2025-02 parquet (Hugging Face)
  fetch.py marco OUT.jsonl N             the first N MS MARCO passages as {"text": ...} lines

Every fetch is cached: a file already in DIR with the expected size is kept, so a rerun costs
nothing and a cloud box can be rebuilt from this script alone. Standard library only. The shapes
a fetched corpus feeds are the ones in benchmarks/columnar-e2e (prep.py turns bgl/taxi into
bulk bodies), benchmarks/newest-n and benchmarks/agg-segment-cache (loghub), and
benchmarks/native-node/text_bench.py (marco). `synthetic` is what `scripts/lr corpus synthetic`
and the CI-sized runs use: the same bytes on every machine, generated in seconds.
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import urllib.request

HF = "https://huggingface.co/datasets"
LOGHUB = HF + "/bolu61/loghub_2/resolve/main/data/{}.txt"
TAXI = HF + "/koorukuroo/yellow_tripdata/resolve/main/yellow_tripdata_2025-0{}.parquet"
MARCO = HF + "/Tevatron/msmarco-passage-corpus/resolve/main/corpus.jsonl.gz"
HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "..", "..", "opensearch-plugin", "e2e")


def download(url, path, limit=None):
    """Stream URL into PATH (the first LIMIT bytes only when given); keeps a finished file."""
    want = limit
    if os.path.isfile(path) and (want is None or os.path.getsize(path) >= want or path.endswith(".done")):
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    n = 0
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20 if want is None else min(1 << 20, want - n))
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
            if want is not None and n >= want:
                break
    os.replace(tmp, path)
    print(f"fetched {path} ({n / 2**20:.0f} MiB)", file=sys.stderr)
    return path


def loghub(out, mb, sets=("bgl", "hdfs", "openssh", "thunderbird")):
    for s in sets:
        download(LOGHUB.format(s), os.path.join(out, f"{s}.txt"), limit=mb * 2**20)


def taxi(out):
    for m in (1, 2):
        download(TAXI.format(m), os.path.join(out, f"yellow_tripdata_2025-0{m}.parquet"))


def marco(out, n):
    """The first N passages of the 8.8M-passage MS MARCO corpus, without downloading the rest."""
    if os.path.isfile(out) and sum(1 for _ in open(out)) >= n:
        return
    import json
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".part"
    with urllib.request.urlopen(MARCO, timeout=120) as r, gzip.GzipFile(fileobj=r) as g, open(tmp, "w") as f:
        for i, line in enumerate(g):
            if i >= n:
                break
            f.write(json.dumps({"text": json.loads(line)["text"]}) + "\n")
    os.replace(tmp, out)
    print(f"wrote {n} passages to {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("dir")
    s.add_argument("docs", type=int)
    s = sub.add_parser("loghub")
    s.add_argument("dir")
    s.add_argument("--mb", type=int, default=250)
    sub.add_parser("bgl").add_argument("dir")
    sub.add_parser("taxi").add_argument("dir")
    s = sub.add_parser("marco")
    s.add_argument("out")
    s.add_argument("n", type=int)
    a = ap.parse_args()
    if a.cmd == "synthetic":
        if os.path.isfile(os.path.join(a.dir, "done-geo")):
            print(f"{a.dir}: already generated")
            return
        shutil.rmtree(a.dir, ignore_errors=True)
        subprocess.run([sys.executable, os.path.join(E2E, "bench_nodes.py"), "gen", a.dir, str(a.docs)], check=True)
        open(os.path.join(a.dir, "done-geo"), "w").close()
    elif a.cmd == "loghub":
        loghub(a.dir, a.mb)
    elif a.cmd == "bgl":
        download(LOGHUB.format("bgl"), os.path.join(a.dir, "bgl.txt"))
    elif a.cmd == "taxi":
        taxi(a.dir)
    elif a.cmd == "marco":
        marco(a.out, a.n)


if __name__ == "__main__":
    main()
