package org.accelerated.lucene.spark;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ArrowColumnVector;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/** One scan: partitions are groups of segments, read as Arrow batches over the C stream interface. */
final class LuceneScan implements Scan, Batch, PartitionReaderFactory {
  private static final long serialVersionUID = 1L;
  private static final ConcurrentHashMap<String, Long> TABLES = new ConcurrentHashMap<>();
  private static final BufferAllocator ALLOCATOR = new RootAllocator();

  private final StructType schema;
  private final Options options;
  private final String request;

  LuceneScan(StructType schema, Options options, String request) {
    this.schema = schema;
    this.options = options;
    this.request = request;
  }

  @Override
  public StructType readSchema() {
    return schema;
  }

  @Override
  public Batch toBatch() {
    return this;
  }

  @Override
  public InputPartition[] planInputPartitions() {
    int segments;
    try {
      segments = segments(table(options));
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    int wanted = options.partitions > 0
        ? options.partitions
        : Math.max(1, Math.min(segments, defaultParallelism()));
    int n = Math.max(1, Math.min(wanted, segments));
    InputPartition[] parts = new InputPartition[n];
    for (int i = 0; i < n; i++) {
      parts[i] = new Part(i, n);
    }
    return parts;
  }

  private static int defaultParallelism() {
    try {
      return org.apache.spark.sql.SparkSession.active().sparkContext().defaultParallelism();
    } catch (RuntimeException e) {
      return Runtime.getRuntime().availableProcessors();
    }
  }

  private static int segments(long table) throws IOException {
    String info = Native.info(table);
    int at = info.indexOf("\"segments\":") + 11;
    int end = at;
    while (end < info.length() && Character.isDigit(info.charAt(end))) {
      end++;
    }
    return Integer.parseInt(info.substring(at, end));
  }

  /** The table, opened once per JVM and index. */
  private static long table(Options o) throws IOException {
    Native.load();
    String key = o.path + "\n" + o.spec.json();
    Long t = TABLES.get(key);
    if (t == null) {
      synchronized (TABLES) {
        t = TABLES.get(key);
        if (t == null) {
          t = Native.open(o.path, o.spec.json());
          TABLES.put(key, t);
        }
      }
    }
    return t;
  }

  @Override
  public PartitionReaderFactory createReaderFactory() {
    return this;
  }

  @Override
  public boolean supportColumnarReads(InputPartition partition) {
    return true;
  }

  @Override
  public PartitionReader<org.apache.spark.sql.catalyst.InternalRow> createReader(InputPartition partition) {
    throw new UnsupportedOperationException("the lucene source reads columnar batches");
  }

  @Override
  public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
    Part p = (Part) partition;
    try {
      long table = table(options);
      ArrowArrayStream stream = ArrowArrayStream.allocateNew(ALLOCATOR);
      try {
        Native.scan(table, request, p.index, p.count, stream.memoryAddress());
      } catch (IOException | RuntimeException e) {
        stream.close();
        throw e;
      }
      ArrowReader reader = Data.importArrayStream(ALLOCATOR, stream);
      return new Reader(reader);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static final class Part implements InputPartition {
    private static final long serialVersionUID = 1L;
    final int index;
    final int count;

    Part(int index, int count) {
      this.index = index;
      this.count = count;
    }
  }

  private static final class Reader implements PartitionReader<ColumnarBatch> {
    private final ArrowReader reader;
    private ColumnarBatch current;

    Reader(ArrowReader reader) {
      this.reader = reader;
    }

    @Override
    public boolean next() {
      VectorSchemaRoot root;
      try {
        if (!reader.loadNextBatch()) {
          return false;
        }
        root = reader.getVectorSchemaRoot();
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
      ColumnVector[] cols = new ColumnVector[root.getFieldVectors().size()];
      int i = 0;
      for (FieldVector v : root.getFieldVectors()) {
        cols[i++] = new ArrowColumnVector(v);
      }
      current = new ColumnarBatch(cols, root.getRowCount());
      return true;
    }

    @Override
    public ColumnarBatch get() {
      return current;
    }

    @Override
    public void close() throws IOException {
      reader.close();
    }
  }
}
