# HNSW benchmarks

Recall and single-thread latency of this port's Lucene HNSW against Java
Lucene on the same index files and against hnswlib on the same data.
Results: [`docs/benchmarks/hnsw.md`](../../docs/benchmarks/hnsw.md).

Data as in [`../ivf/README.md`](../ivf/README.md) (`prepare.py` writes
`<prefix>.base.fbin`, `.query.fbin`, `.gt.ibin`).

```
# Rust: build once into DIR (--reuse keeps it), then sweep ef
cargo run --release -p lucene-search --example knn_formats -- /tmp/data/glove100 cosine \
    --formats hnsw,hnsw-int7 --ef 10,40,100,320 --dir DIR --reuse --queries 2000

# Java Lucene on the index Rust wrote (DIR/knn-formats-hnsw), and a Java build
CP=fixtures/.jars/lucene-core-10.5.0.jar
javac -cp $CP -d /tmp/kb benchmarks/hnsw/KnnBench.java benchmarks/hnsw/KnnBuild.java
java --add-modules jdk.incubator.vector -cp $CP:/tmp/kb KnnBench DIR/knn-formats-hnsw /tmp/data/glove100 10,40,100,320 2000
java --add-modules jdk.incubator.vector -Xmx6g -cp $CP:/tmp/kb KnnBuild /tmp/java-idx /tmp/data/glove100

# hnswlib (pip install hnswlib numpy), M 16, ef_construction 100, one thread
python3 benchmarks/hnsw/hnswlib_bench.py /tmp/data/glove100 10,40,100,320 2000
```

`KnnBench` reads ground truth by doc id, so it needs an unmerged
single-segment index (doc id = base ordinal), which `knn_formats` writes by
default.
