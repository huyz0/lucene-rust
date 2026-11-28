# lucene-spark

A Spark 4.0 DataSource V2 source over an index: column pruning, comparisons and
`LIMIT` run inside the index scan, and batches reach Spark as Arrow columnar
batches over the C data interface. See
[the index as an Arrow source](../../docs/arrow-source.md).

```
mvn -f connectors/spark package
cargo build --release -p lucene-ffi --no-default-features --features crypto,arrow-sql,spark

spark-submit --jars connectors/spark/target/lucene-spark-0.1.0.jar,arrow-c-data-18.1.0.jar \
  --conf spark.driver.extraJavaOptions=-Dlucene.native.library=$PWD/target/release/liblucene_ffi.so \
  --conf spark.executor.extraJavaOptions=-Dlucene.native.library=$PWD/target/release/liblucene_ffi.so job.py
```

```python
df = (spark.read.format("lucene").option("path", "/data/index").option("table", "logs")
      .option("columns", "id:key,ts:long,status:long,host:keyword").load())
```

Options: `path`, `columns` (`name:kind`, kinds `key`, `long`, `keyword`, `text`,
`vector:dim`), `table`, `partitions` (default: one per segment up to the default
parallelism), `batch_rows`. Pushed down: required columns, `=`, `!=`, `<`, `<=`,
`>`, `>=`, `IN`, `IS NULL`, `IS NOT NULL` on `long` and `keyword` columns (`=`
and `IN` on `key` columns), and `LIMIT`. Spark applies the rest.

`e2e.py` runs the pushed-down and plain queries against the same data as
Parquet and checks the answers are equal (`SPARK_HOME` of a Spark 4.0 install).
