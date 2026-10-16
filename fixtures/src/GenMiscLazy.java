import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.misc.document.LazyDocument;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M14 T14.3 `misc`: {@link LazyDocument}. Writes {@code misc/lazy}, an index of stored fields only
 * (two commits, so two segments), and {@code misc/lazy.tsv}, which records for every document how
 * a {@code LazyDocument} behaves when its fields are handed out in two groups and loaded on use:
 *
 * <pre>
 * L  doc  stage  field  idx  loaded       whether hasBeenLoaded() after the stage
 * V  doc  field  idx  string  number  binary   what the value reads as ("-" for null)
 * </pre>
 *
 * Stages: 1 = group A handed out; 2 = A's first value read; 3 = group B handed out; 4 = B's first
 * value read; 5 = every value read.
 *
 * <p>Read by {@code crates/lucene-search/tests/misc_fixtures.rs}.
 */
public class GenMiscLazy {

  static final String[] GROUP_A = {"s", "i", "b"};
  static final String[] GROUP_B = {"l", "f", "d", "other"};

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    Files.createDirectories(out);
    Path index = out.resolve("lazy");
    Random rnd = new Random(1465L);
    try (Directory dir = FSDirectory.open(index);
        IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
      for (int commit = 0; commit < 2; commit++) {
        for (int d = 0; d < 8; d++) {
          Document doc = new Document();
          for (int n = rnd.nextInt(4), k = 0; k < n; k++) doc.add(new StoredField("s", "sv" + rnd.nextInt(100)));
          for (int n = rnd.nextInt(3), k = 0; k < n; k++) doc.add(new StoredField("i", rnd.nextInt(1000) - 500));
          for (int n = rnd.nextInt(3), k = 0; k < n; k++) doc.add(new StoredField("l", rnd.nextLong() >> 8));
          for (int n = rnd.nextInt(3), k = 0; k < n; k++) doc.add(new StoredField("f", rnd.nextInt(1000) / 8f));
          for (int n = rnd.nextInt(3), k = 0; k < n; k++) doc.add(new StoredField("d", rnd.nextInt(100000) / 16d));
          for (int n = rnd.nextInt(3), k = 0; k < n; k++) {
            byte[] b = new byte[1 + rnd.nextInt(4)];
            rnd.nextBytes(b);
            doc.add(new StoredField("b", new BytesRef(b)));
          }
          if (rnd.nextBoolean()) doc.add(new StoredField("other", "o" + rnd.nextInt(10)));
          w.addDocument(doc);
        }
        w.commit();
      }
    }

    try (Directory dir = FSDirectory.open(index);
        IndexReader reader = DirectoryReader.open(dir);
        PrintWriter tsv =
            new PrintWriter(Files.newBufferedWriter(out.resolve("lazy.tsv"), StandardCharsets.UTF_8))) {
      FieldInfos infos = FieldInfos.getMergedFieldInfos(reader);
      for (int doc = 0; doc < reader.maxDoc(); doc++) {
        Document real = reader.storedFields().document(doc);
        LazyDocument lazy = new LazyDocument(reader, doc);
        // field -> the lazy fields handed out for it
        java.util.LinkedHashMap<String, List<IndexableField>> handed = new java.util.LinkedHashMap<>();
        hand(lazy, real, infos, GROUP_A, handed);
        stage(tsv, doc, 1, handed);
        read(handed, GROUP_A[0], 0);
        stage(tsv, doc, 2, handed);
        hand(lazy, real, infos, GROUP_B, handed);
        stage(tsv, doc, 3, handed);
        read(handed, GROUP_B[0], 0);
        stage(tsv, doc, 4, handed);
        for (var e : handed.entrySet()) {
          for (int i = 0; i < e.getValue().size(); i++) {
            IndexableField f = e.getValue().get(i);
            String s = f.stringValue();
            Number n = f.numericValue();
            BytesRef b = f.binaryValue();
            tsv.printf("V\t%d\t%s\t%d\t%s\t%s\t%s%n", doc, e.getKey(), i, s == null ? "-" : s, num(n), hex(b));
          }
        }
        stage(tsv, doc, 5, handed);
      }
    }
  }

  static void hand(
      LazyDocument lazy,
      Document real,
      FieldInfos infos,
      String[] names,
      java.util.LinkedHashMap<String, List<IndexableField>> handed) {
    for (String name : names) {
      int values = real.getFields(name).length;
      FieldInfo fi = infos.fieldInfo(name);
      List<IndexableField> list = new ArrayList<>();
      for (int i = 0; i < values; i++) list.add(lazy.getField(fi));
      handed.put(name, list);
    }
  }

  /** Reads the {@code idx}th value of {@code name}, when the document has one. */
  static void read(java.util.LinkedHashMap<String, List<IndexableField>> handed, String name, int idx) {
    List<IndexableField> list = handed.get(name);
    if (list != null && idx < list.size()) list.get(idx).stringValue();
  }

  static void stage(
      PrintWriter tsv, int doc, int stage, java.util.LinkedHashMap<String, List<IndexableField>> handed) {
    for (var e : handed.entrySet()) {
      for (int i = 0; i < e.getValue().size(); i++) {
        LazyDocument.LazyField f = (LazyDocument.LazyField) e.getValue().get(i);
        tsv.printf("L\t%d\t%d\t%s\t%d\t%d%n", doc, stage, e.getKey(), i, f.hasBeenLoaded() ? 1 : 0);
      }
    }
  }

  static String num(Number n) {
    if (n == null) return "-";
    if (n instanceof Integer) return "I:" + n;
    if (n instanceof Long) return "J:" + n;
    if (n instanceof Float f) return "F:" + Float.floatToRawIntBits(f);
    if (n instanceof Double d) return "D:" + Double.doubleToRawLongBits(d);
    throw new IllegalStateException(n.getClass().toString());
  }

  static String hex(BytesRef b) {
    if (b == null) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < b.length; i++) sb.append(String.format("%02x", b.bytes[b.offset + i] & 0xFF));
    return sb.length() == 0 ? "+" : sb.toString();
  }
}
