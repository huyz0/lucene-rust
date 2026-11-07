#!/usr/bin/env python3
"""Writes the two stored-fields benchmark corpora, one JSON `_source` per line.

  europarl.json  every line of Lucene's test-framework europarl.lines.txt.gz (real text in
                 eleven languages) as {"id", "date", "title", "body"}: text-heavy documents.
  logs.json      seeded synthetic web-server log events shaped like OpenSearch Benchmark's
                 http_logs track: small, repetitive JSON documents.

Usage: make-corpus.py EUROPARL_LINES_GZ OUT_DIR [LOG_DOCS]
"""
import gzip
import json
import os
import random
import sys


def main():
    src, out = sys.argv[1], sys.argv[2]
    log_docs = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    os.makedirs(out, exist_ok=True)
    with gzip.open(src, "rt", encoding="utf-8") as f, open(os.path.join(out, "europarl.json"), "w") as o:
        for n, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            title, _, body = parts[2].partition("  ")
            o.write(json.dumps({"id": n, "date": parts[1], "title": title.strip(), "body": body.strip()},
                               ensure_ascii=False) + "\n")
    r = random.Random(20261106)
    paths = [f"/{d}/{w}{i}.{e}" for d in ("images", "english", "french", "docs", "api/v1")
             for w in ("index", "logo", "item", "user", "search") for i in range(40)
             for e in ("html", "gif", "jpg", "json")]
    agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Mozilla/5.0 (X11; Linux x86_64)",
              "curl/8.4.0", "Googlebot/2.1 (+http://www.google.com/bot.html)", "python-requests/2.31"]
    ips = [f"{r.randrange(1, 255)}.{r.randrange(256)}.{r.randrange(256)}.{r.randrange(256)}" for _ in range(5000)]
    ts = 893964617
    with open(os.path.join(out, "logs.json"), "w") as o:
        for _ in range(log_docs):
            ts += r.choice((0, 0, 1, 1, 2, 5))
            status = r.choices((200, 304, 404, 500, 206), (80, 10, 6, 1, 3))[0]
            doc = {"@timestamp": ts, "clientip": ips[int(r.paretovariate(1.2)) % len(ips)],
                   "request": f"GET {paths[int(r.paretovariate(1.1)) % len(paths)]} HTTP/1.1",
                   "status": status, "size": 0 if status == 304 else r.randrange(200, 60000),
                   "agent": agents[r.randrange(len(agents))]}
            o.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
