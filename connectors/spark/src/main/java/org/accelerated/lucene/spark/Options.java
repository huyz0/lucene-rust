package org.accelerated.lucene.spark;

import java.io.Serializable;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/** The reader's options, serializable to the executors. */
final class Options implements Serializable {
  private static final long serialVersionUID = 1L;

  final String path;
  final Spec spec;
  final int partitions;
  final int batchRows;

  private Options(String path, Spec spec, int partitions, int batchRows) {
    this.path = path;
    this.spec = spec;
    this.partitions = partitions;
    this.batchRows = batchRows;
  }

  static Options of(CaseInsensitiveStringMap o) {
    String path = o.get("path");
    String columns = o.get("columns");
    if (path == null || columns == null) {
      throw new IllegalArgumentException("options `path` and `columns` are required");
    }
    return new Options(
        path,
        new Spec(o.getOrDefault("table", "t"), columns),
        o.getInt("partitions", 0),
        o.getInt("batch_rows", 0));
  }
}
