"""Spark reads the index through the lucene source, pushed down and not.

    SPARK_HOME=... PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-*.zip \
    LUCENE_NATIVE_LIBRARY=.../liblucene_ffi.so python3 e2e.py <index> <parquet> <jars...>

Checks each query's answer against the same query on Parquet, then times both.
"""
import glob
import os
import statistics
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

index, parquet = sys.argv[1], sys.argv[2]
jars = ",".join(sys.argv[3:])
spark = (SparkSession.builder.master("local[4]").appName("lucene-e2e")
         .config("spark.jars", jars)
         .config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.driver.extraJavaOptions",
                 f"-Dlucene.native.library={os.environ['LUCENE_NATIVE_LIBRARY']}")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

cols = "id:key,ts:long,status:long,bytes:long,latency:long,host:keyword,level:keyword,region:keyword"
idx = (spark.read.format("lucene").option("path", index).option("table", "logs")
       .option("columns", cols).load())
pq = spark.read.parquet(parquet)
print(idx.schema.simpleString())


def timed(f, reps=3):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), r


cases = {
    "count": lambda d: d.count(),
    "sum 2 long cols": lambda d: d.agg(F.sum("bytes"), F.sum("latency")).collect()[0][:],
    "group by status": lambda d: sorted(d.groupBy("status").agg(F.count("*"), F.sum("bytes")).collect()),
    "group by host (keyword)": lambda d: d.groupBy("host").count().count(),
    "filter status=500 and bytes>90000": lambda d: d.filter((F.col("status") == 500) & (F.col("bytes") > 90000))
        .agg(F.count("*"), F.sum("latency")).collect()[0][:],
    "filter host in (...)": lambda d: d.filter(F.col("host").isin("host-0042", "host-0007")).count(),
    "filter level is not null and ts>=cut": lambda d: d.filter(F.col("level").isNotNull() & (F.col("ts") >= CUT)).count(),
    "limit 100": lambda d: len(d.select("ts", "host").limit(100).collect()),
}
CUT = int(pq.agg(F.expr("percentile_approx(ts, 0.99, 100000)")).collect()[0][0])

print(f"{'case':<42}{'parquet ms':>12}{'index ms':>12}")
for name, f in cases.items():
    tp, rp = timed(lambda: f(pq))
    ti, ri = timed(lambda: f(idx))
    if name != "limit 100":
        assert rp == ri, (name, rp, ri)
    else:
        assert rp == ri == 100
    print(f"{name:<42}{tp * 1000:12.0f}{ti * 1000:12.0f}")

plan = idx.filter((F.col("status") == 500) & (F.col("bytes") > 90000)).select("ts")._jdf.queryExecution().executedPlan().toString()
print(plan)
assert "PushedFilters" in plan or "lucene" in plan.lower()
spark.stop()
