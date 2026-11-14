"""Writes real-data bulk bodies for columnar_e2e.py.

  python3 -I prep.py taxi PARQUET_DIR OUT_DIR DOCS    NYC yellow taxi, 2025 months in order
  python3 -I prep.py bgl BGL_TXT OUT_DIR DOCS         Loghub BGL log lines

taxi: koorukuroo/yellow_tripdata (Hugging Face) yellow_tripdata_2025-01.parquet and -02.
bgl:  bolu61/loghub_2 data/bgl.txt (Hugging Face).
Documents keep the file's order (time order for bgl; pickup order for taxi, which is
not time order within a month). DOCS caps the count (0 = all)."""
import glob
import json
import os
import sys

PER_FILE = 5000


def write(out, docs_iter, limit):
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "bulk-*.ndjson")):
        os.remove(f)
    buf, n, files = [], 0, 0
    for d in docs_iter:
        buf.append('{"index":{}}')
        buf.append(json.dumps(d, separators=(",", ":")))
        n += 1
        if len(buf) == PER_FILE * 2:
            open(os.path.join(out, "bulk-%05d.ndjson" % files), "w").write("\n".join(buf) + "\n")
            buf, files = [], files + 1
        if limit and n >= limit:
            break
    if buf:
        open(os.path.join(out, "bulk-%05d.ndjson" % files), "w").write("\n".join(buf) + "\n")
        files += 1
    print(f"{n} documents in {files} files under {out}")


def taxi(src, out, limit):
    import pyarrow.parquet as pq

    def gen():
        for path in sorted(glob.glob(os.path.join(src, "*.parquet"))):
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=50_000, columns=[
                    "tpep_pickup_datetime", "PULocationID", "passenger_count", "trip_distance",
                    "fare_amount", "tip_amount", "total_amount"]):
                c = batch.to_pydict()
                for i in range(batch.num_rows):
                    ts = c["tpep_pickup_datetime"][i]
                    d = {"pickup": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                         "pu_location": str(c["PULocationID"][i]),
                         "fare": c["fare_amount"][i], "tip": c["tip_amount"][i],
                         "distance": c["trip_distance"][i], "total": c["total_amount"][i]}
                    if c["passenger_count"][i] is not None:  # nulls stay missing: a sparse field
                        d["passengers"] = int(c["passenger_count"][i])
                    yield {k: v for k, v in d.items() if v is not None}
    write(out, gen(), limit)


def bgl(src, out, limit):
    def gen():
        with open(src, encoding="iso-8859-1") as fh:
            for line in fh:
                line = line.rstrip("\n")
                p = line.split(" ", 9)
                if len(p) < 9 or not p[1].isdigit():
                    continue
                frac = p[4].rsplit(".", 1)[-1]
                micros = int(frac) if frac.isdigit() else 0
                yield {"@timestamp": int(p[1]) * 1000 + micros // 1000, "host": p[3],
                       "component": p[7], "level": p[8], "bytes": len(line),
                       "message": p[9] if len(p) > 9 else ""}
    write(out, gen(), limit)


{"taxi": taxi, "bgl": bgl}[sys.argv[1]](sys.argv[2], sys.argv[3], int(sys.argv[4]))
