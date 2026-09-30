import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StoredValue;
import org.apache.lucene.document.column.BinaryColumn;
import org.apache.lucene.document.column.BytesRefValuesCursor;
import org.apache.lucene.document.column.Column;
import org.apache.lucene.document.column.ColumnBatch;
import org.apache.lucene.document.column.DictionaryColumn;
import org.apache.lucene.document.column.LongColumn;
import org.apache.lucene.document.column.LongTupleCursor;
import org.apache.lucene.document.column.LongValuesCursor;
import org.apache.lucene.document.column.ObjectTupleCursor;
import org.apache.lucene.document.column.OrdinalsCursor;
import org.apache.lucene.document.column.OrdinalsTupleCursor;
import org.apache.lucene.document.column.TokenStreamColumn;
import org.apache.lucene.document.column.VectorColumn;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * The columnar batch API ({@code IndexWriter.addBatch}, {@code org.apache.lucene.document.column})
 * recorded for crates/lucene-search/tests/document_fields_fixtures.rs: two batches of every column
 * kind, dense and sparse, written by Lucene into {@code document_columns/index}, and {@code
 * columns.tsv}, the batches' columns as data -- which the Rust test turns into the same batches,
 * writes with this port's {@code add_batch}, and compares with Lucene's index field by field.
 *
 * <p>{@code columns.tsv}: a {@code batch <numDocs>} line, then one line per column: kind ({@code
 * L} long, {@code B} binary, {@code D} dictionary, {@code T} token stream, {@code V} vector), name,
 * density, a field type recipe name (see {@link #type}), a kind-specific extra ({@code
 * NumericKind}, stored type, the dictionary as hex, or {@code -}), then the cells, {@code
 * doc:value} (hex bytes for binary, the ordinal for a dictionary, the text for a token stream,
 * comma-separated floats for a vector).
 */
public class GenDocumentColumns {
  static final HexFormat HEX = HexFormat.of();
  static final Analyzer ANALYZER = new StandardAnalyzer();
  static final String[] WORDS = {"red", "green", "blue", "cyan", "magenta", "yellow"};

  /** The field type recipes both sides share. */
  static FieldType type(String recipe) {
    FieldType t = new FieldType();
    switch (recipe) {
      case "int_point_sndv_stored" -> {
        t.setDimensions(1, 4);
        t.setDocValuesType(DocValuesType.SORTED_NUMERIC);
        t.setStored(true);
      }
      case "numeric_dv_skip" -> {
        t.setDocValuesType(DocValuesType.NUMERIC);
        t.setDocValuesSkipIndexType(DocValuesSkipIndexType.RANGE);
      }
      case "float_point_sndv_stored" -> {
        t.setDimensions(1, 4);
        t.setDocValuesType(DocValuesType.SORTED_NUMERIC);
        t.setStored(true);
      }
      case "long_point_stored" -> {
        t.setDimensions(1, 8);
        t.setStored(true);
      }
      case "keyword_stored" -> {
        t.setIndexOptions(IndexOptions.DOCS);
        t.setOmitNorms(true);
        t.setTokenized(false);
        t.setDocValuesType(DocValuesType.SORTED_SET);
        t.setStored(true);
      }
      case "binary_dv" -> t.setDocValuesType(DocValuesType.BINARY);
      case "point_2x2" -> t.setDimensions(2, 2);
      case "text_stored" -> {
        t.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
        t.setTokenized(true);
        t.setStored(true);
      }
      case "sorted_dv" -> t.setDocValuesType(DocValuesType.SORTED);
      case "sorted_set_dv" -> t.setDocValuesType(DocValuesType.SORTED_SET);
      case "text_offsets" -> {
        t.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
        t.setTokenized(true);
      }
      case "vector3" -> t.setVectorAttributes(3, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);
      default -> throw new IllegalArgumentException(recipe);
    }
    t.freeze();
    return t;
  }

  record Cell(int doc, String value) {}

  record Spec(char kind, String name, Column.Density density, String recipe, String extra, List<Cell> cells) {
    String line() {
      StringBuilder sb = new StringBuilder();
      sb.append(kind).append('\t').append(name).append('\t').append(density).append('\t')
          .append(recipe).append('\t').append(extra);
      for (Cell c : cells) sb.append('\t').append(c.doc).append(':').append(c.value);
      return sb.toString();
    }
  }

  static Column column(Spec s) {
    FieldType t = type(s.recipe);
    List<Cell> cells = s.cells;
    switch (s.kind) {
      case 'L':
        return new LongColumn(s.name, t, s.density, LongColumn.NumericKind.valueOf(s.extra)) {
          @Override
          public LongTupleCursor tuples() {
            return new LongTupleCursor() {
              int i = -1;

              @Override
              public int nextDoc() {
                return ++i < cells.size() ? cells.get(i).doc : DocIdSetIterator.NO_MORE_DOCS;
              }

              @Override
              public long longValue() {
                return Long.parseLong(cells.get(i).value);
              }
            };
          }

          @Override
          public LongValuesCursor values() {
            return new LongValuesCursor(cells.size()) {
              int i = 0;

              @Override
              public long nextLong() {
                return Long.parseLong(cells.get(i++).value);
              }
            };
          }
        };
      case 'B':
        return new BinaryColumn(s.name, t, s.density) {
          @Override
          public StoredValue.Type storedType() {
            return StoredValue.Type.valueOf(s.extra);
          }

          @Override
          public ObjectTupleCursor<BytesRef> tuples() {
            return new ObjectTupleCursor<>() {
              int i = -1;

              @Override
              public int nextDoc() {
                return ++i < cells.size() ? cells.get(i).doc : DocIdSetIterator.NO_MORE_DOCS;
              }

              @Override
              public BytesRef value() {
                return new BytesRef(HEX.parseHex(cells.get(i).value));
              }
            };
          }

          @Override
          public BytesRefValuesCursor values() {
            return new BytesRefValuesCursor(cells.size()) {
              int i = 0;

              @Override
              public BytesRef nextValue() {
                return new BytesRef(HEX.parseHex(cells.get(i++).value));
              }
            };
          }
        };
      case 'D':
        {
          List<BytesRef> dict = new ArrayList<>();
          for (String h : s.extra.split(",")) dict.add(new BytesRef(HEX.parseHex(h)));
          return new DictionaryColumn(s.name, t, s.density, dict) {
            @Override
            public OrdinalsTupleCursor tuples() {
              return new OrdinalsTupleCursor() {
                int i = -1;

                @Override
                public int nextDoc() {
                  return ++i < cells.size() ? cells.get(i).doc : DocIdSetIterator.NO_MORE_DOCS;
                }

                @Override
                public int ordValue() {
                  return Integer.parseInt(cells.get(i).value);
                }
              };
            }

            @Override
            public OrdinalsCursor values() {
              return new OrdinalsCursor(cells.size()) {
                int i = 0;

                @Override
                public int nextOrd() {
                  return Integer.parseInt(cells.get(i++).value);
                }
              };
            }
          };
        }
      case 'T':
        return new TokenStreamColumn(s.name, t, s.density) {
          @Override
          public ObjectTupleCursor<TokenStream> tuples() {
            return new ObjectTupleCursor<>() {
              int i = -1;

              @Override
              public int nextDoc() {
                return ++i < cells.size() ? cells.get(i).doc : DocIdSetIterator.NO_MORE_DOCS;
              }

              @Override
              public TokenStream value() {
                return ANALYZER.tokenStream(s.name, cells.get(i).value.replace('_', ' '));
              }
            };
          }
        };
      case 'V':
        return new VectorColumn<float[]>(s.name, t, s.density) {
          @Override
          public ObjectTupleCursor<float[]> tuples() {
            return new ObjectTupleCursor<>() {
              int i = -1;

              @Override
              public int nextDoc() {
                return ++i < cells.size() ? cells.get(i).doc : DocIdSetIterator.NO_MORE_DOCS;
              }

              @Override
              public float[] value() {
                String[] p = cells.get(i).value.split(",");
                float[] v = new float[p.length];
                for (int k = 0; k < p.length; k++) v[k] = Float.parseFloat(p[k]);
                return v;
              }
            };
          }
        };
      default:
        throw new IllegalArgumentException(String.valueOf(s.kind));
    }
  }

  /** Cells for every doc (dense) or a random subset, some repeated (sparse, multi-valued). */
  static List<Cell> cells(Random r, int numDocs, boolean dense, boolean multi, java.util.function.Supplier<String> value) {
    List<Cell> out = new ArrayList<>();
    for (int d = 0; d < numDocs; d++) {
      if (dense || r.nextInt(3) != 0) {
        out.add(new Cell(d, value.get()));
        if (!dense && multi && r.nextInt(4) == 0) out.add(new Cell(d, value.get()));
      }
    }
    return out;
  }

  static String hex(byte[] b) {
    return HEX.formatHex(b);
  }

  static String words(Random r) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0, n = 1 + r.nextInt(5); i < n; i++) {
      if (i > 0) sb.append('_');
      sb.append(WORDS[r.nextInt(WORDS.length)]);
    }
    return sb.toString();
  }

  static List<Spec> batch(Random r, int numDocs) {
    List<Spec> specs = new ArrayList<>();
    Column.Density dense = Column.Density.DENSE, sparse = Column.Density.SPARSE;
    specs.add(new Spec('L', "n", dense, "int_point_sndv_stored", "INT",
        cells(r, numDocs, true, false, () -> Integer.toString(r.nextInt(201) - 100))));
    specs.add(new Spec('L', "l", sparse, "numeric_dv_skip", "LONG",
        cells(r, numDocs, false, false, () -> Long.toString(r.nextLong() >> r.nextInt(64)))));
    specs.add(new Spec('L', "f", sparse, "float_point_sndv_stored", "FLOAT",
        cells(r, numDocs, false, true,
            () -> Integer.toString(NumericUtils.floatToSortableInt((r.nextInt(401) - 200) / 4f)))));
    specs.add(new Spec('L', "dd", dense, "long_point_stored", "DOUBLE",
        cells(r, numDocs, true, false,
            () -> Long.toString(NumericUtils.doubleToSortableLong((r.nextInt(801) - 400) / 8.0)))));
    specs.add(new Spec('B', "kw", sparse, "keyword_stored", "STRING",
        cells(r, numDocs, false, true,
            () -> hex(WORDS[r.nextInt(WORDS.length)].getBytes(StandardCharsets.UTF_8)))));
    specs.add(new Spec('B', "b", dense, "binary_dv", "BINARY",
        cells(r, numDocs, true, false, () -> {
          byte[] b = new byte[r.nextInt(6)];
          r.nextBytes(b);
          return hex(b);
        })));
    specs.add(new Spec('B', "bp", sparse, "point_2x2", "BINARY",
        cells(r, numDocs, false, true, () -> {
          byte[] b = new byte[4];
          r.nextBytes(b);
          return hex(b);
        })));
    specs.add(new Spec('B', "txt", sparse, "text_stored", "STRING",
        cells(r, numDocs, false, false, () -> hex(words(r).replace('_', ' ').getBytes(StandardCharsets.UTF_8)))));
    StringBuilder dict = new StringBuilder();
    for (int i = 0; i < WORDS.length; i++) {
      if (i > 0) dict.append(',');
      dict.append(hex(WORDS[i].getBytes(StandardCharsets.UTF_8)));
    }
    specs.add(new Spec('D', "cat", dense, "sorted_dv", dict.toString(),
        cells(r, numDocs, true, false, () -> Integer.toString(r.nextInt(WORDS.length)))));
    specs.add(new Spec('D', "tags", sparse, "sorted_set_dv", dict.toString(),
        cells(r, numDocs, false, true, () -> Integer.toString(r.nextInt(WORDS.length)))));
    specs.add(new Spec('T', "ts", sparse, "text_offsets", "-",
        cells(r, numDocs, false, false, () -> words(r))));
    specs.add(new Spec('V', "vec", sparse, "vector3", "-",
        cells(r, numDocs, false, false,
            () -> (r.nextInt(21) - 10) / 2f + "," + (r.nextInt(21) - 10) / 2f + "," + (r.nextInt(21) - 10) / 2f)));
    if (numDocs == 35) {
      // Columns that yield no value in this batch: Lucene still registers each field's FieldInfo in
      // the segment (processBatch initializes every column's field before reading a cell).
      specs.add(new Spec('L', "none_l", sparse, "numeric_dv_skip", "LONG", List.of()));
      specs.add(new Spec('B', "none_kw", sparse, "keyword_stored", "STRING", List.of()));
      specs.add(new Spec('T', "none_ts", sparse, "text_offsets", "-", List.of()));
    }
    return specs;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("document_columns");
    if (Files.exists(root)) {
      try (Stream<Path> s = Files.walk(root)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(20261001L);
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(ANALYZER);
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int numDocs : new int[] {60, 35}) {
          List<Spec> specs = batch(r, numDocs);
          out.append("batch\t").append(numDocs).append('\n');
          for (Spec s : specs) out.append(s.line()).append('\n');
          List<Column> columns = new ArrayList<>();
          for (Spec s : specs) columns.add(column(s));
          w.addBatch(
              new ColumnBatch() {
                @Override
                public int numDocs() {
                  return numDocs;
                }

                @Override
                public Iterable<Column> columns() {
                  return columns;
                }
              });
          w.commit();
        }
      }
    }
    Files.writeString(root.resolve("columns.tsv"), out);
  }
}
