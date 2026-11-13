"""Exports doc-values-shaped columns (raw little-endian i64, as OpenSearch stores them:
dates as epoch ms, doubles as NumericUtils sortable longs, keywords as ordinals of the sorted
dictionary) from NYC yellow taxi 2025-01/02 parquet and Loghub BGL.
  python3 -I prep.py DATA_DIR OUT_DIR
"""
import os, struct, sys
import numpy as np
import pyarrow.parquet as pq

data, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

def sortable(d):
    b = d.astype(np.float64).view(np.int64)
    return b ^ ((b >> 63) & np.int64(0x7FFFFFFFFFFFFFFF))

def ords(strs):
    u, inv = np.unique(np.asarray(strs), return_inverse=True)
    return inv.astype(np.int64), len(u)

def save(name, arr):
    arr = np.ascontiguousarray(arr, dtype="<i8")
    arr.tofile(os.path.join(out, name + ".i64"))
    print(f"{name}: n={len(arr)} min={arr.min()} max={arr.max()}", flush=True)

# --- taxi
t = pq.read_table(os.path.join(data, "yellow_tripdata_2025-01.parquet"))
t2 = pq.read_table(os.path.join(data, "yellow_tripdata_2025-02.parquet"))
import pyarrow as pa
t = pa.concat_tables([t, t2.cast(t.schema)])
cols = {}
cols["ts"] = t["tpep_pickup_datetime"].cast(pa.timestamp("ms")).cast(pa.int64()).to_numpy(zero_copy_only=False)
for c, n in [("fare_amount", "fare"), ("tip_amount", "tip"), ("trip_distance", "distance"),
             ("total_amount", "total")]:
    cols[n] = sortable(t[c].fill_null(0.0).to_numpy(zero_copy_only=False))
cols["passengers"] = t["passenger_count"].fill_null(0).cast(pa.int64()).to_numpy(zero_copy_only=False)
cols["pu_location"] = t["PULocationID"].cast(pa.int64()).to_numpy(zero_copy_only=False)
loc_ord, nloc = ords(t["PULocationID"].cast(pa.string()).to_numpy(zero_copy_only=False))
cols["pu_location_kw"] = loc_ord
for k, v in cols.items():
    save("taxi_" + k, v)
order = np.argsort(cols["ts"], kind="stable")
for k, v in cols.items():
    save("taxisorted_" + k, v[order])

# --- BGL
ts, host, level, comp, nbytes = [], [], [], [], []
with open(os.path.join(data, "bgl.txt"), encoding="iso-8859-1") as fh:
    for line in fh:
        line = line.rstrip("\n")
        p = line.split(" ", 9)
        if len(p) < 9 or not p[1].isdigit():
            continue
        frac = p[4].rsplit(".", 1)[-1]
        micros = int(frac) if frac.isdigit() else 0
        ts.append(int(p[1]) * 1000 + micros // 1000)
        host.append(p[3]); comp.append(p[7]); level.append(p[8]); nbytes.append(len(line))
save("bgl_ts", np.array(ts, dtype=np.int64))
for name, vals in [("host", host), ("level", level), ("component", comp)]:
    o, n = ords(vals)
    print(f"bgl_{name}: {n} distinct")
    save("bgl_" + name, o)
save("bgl_bytes", np.array(nbytes, dtype=np.int64))
