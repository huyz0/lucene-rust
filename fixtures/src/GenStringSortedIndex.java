import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BinarySortField;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.SortedSetSortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * The physical document order real Lucene gives the three index sorts whose
 * key is a byte string: {@code SortField.Type.STRING} over SORTED,
 * {@code SortedSetSortField} (all four selectors) over SORTED_SET, and
 * {@code BinarySortField} over BINARY -- each with {@code reverse}, every
 * missing-value form, and a second tier.
 *
 * <p>For each of eight sorts one {@code IndexWriter} indexes the same 120
 * documents in three committed batches (three sort-on-flush segments),
 * deletes every seventh document and force-merges the rest into one segment
 * (a {@code MultiSorter} k-way merge across segments whose key ranges
 * overlap completely). The manifest records the documents and, per sort, the
 * id order of each flushed segment and of the merged one; the merged index
 * itself is kept so this port's {@code CheckIndex} can verify a Java-sorted
 * segment of each kind.
 *
 * <p>The values are chosen for the ways a byte comparison goes wrong: bytes
 * above {@code 0x7f} (unsigned, so {@code 0xff} is the largest), multi-byte
 * UTF-8, the empty value, prefixes ({@code "ab"} < {@code "abc"}), duplicate
 * values inside one document's SORTED_SET, and documents missing each field.
 */
public class GenStringSortedIndex {
  static final int BATCHES = 3;
  static final int PER_BATCH = 40;

  static final String[] NAMES = {
    "apple", "Apple", "äpfel", "zebra", "z", "", "ÿ", "日本", "ab", "abc", "b"
  };
  static final String[] TAGS = {
    "t0", "t1", "t10", "t2", "u", "é", "a", "zz", "m", "", "bÿ", "k"
  };
  static final byte[] BLOB_BYTES = {0x00, 0x01, 0x7f, (byte) 0x80, (byte) 0xff, 'a'};

  /** One document: null means the field is absent. */
  record Row(String id, byte[] name, List<byte[]> tags, byte[] blob, long seq) {}

  record Config(String name, Sort sort) {}

  static List<Config> configs() {
    List<Config> c = new ArrayList<>();
    SortField s;

    s = new SortField("name", SortField.Type.STRING);
    s.setMissingValue(SortField.STRING_LAST);
    c.add(new Config("string_asc_last", new Sort(s, new SortField("seq", SortField.Type.LONG))));

    c.add(new Config("string_desc", new Sort(
        new SortField("name", SortField.Type.STRING, true),
        new SortField("seq", SortField.Type.LONG, true))));

    c.add(new Config("set_min", new Sort(
        new SortedSetSortField("tags", false),
        new SortField("seq", SortField.Type.LONG))));

    s = new SortedSetSortField("tags", true, SortedSetSelector.Type.MAX);
    s.setMissingValue(SortField.STRING_LAST);
    c.add(new Config("set_max_desc_last", new Sort(s)));

    s = new SortedSetSortField("tags", false, SortedSetSelector.Type.MIDDLE_MIN);
    s.setMissingValue(SortField.STRING_FIRST);
    c.add(new Config("set_middle_min_then_blob", new Sort(s, new BinarySortField("blob", false))));

    c.add(new Config("set_middle_max_desc_then_name", new Sort(
        new SortedSetSortField("tags", true, SortedSetSelector.Type.MIDDLE_MAX),
        new SortField("name", SortField.Type.STRING))));

    c.add(new Config("binary_desc_first", new Sort(
        new BinarySortField("blob", true, SortField.STRING_FIRST),
        new SortField("seq", SortField.Type.LONG))));

    c.add(new Config("binary_last", new Sort(
        new BinarySortField("blob", false, SortField.STRING_LAST))));
    return c;
  }

  static List<Row> rows() {
    Random r = new Random(0x5eedL);
    List<Row> rows = new ArrayList<>();
    for (int i = 0; i < BATCHES * PER_BATCH; i++) {
      byte[] name = r.nextInt(6) == 0 ? null : NAMES[r.nextInt(NAMES.length)].getBytes(StandardCharsets.UTF_8);
      List<byte[]> tags = null;
      int nTags = r.nextInt(5); // 0 = absent
      if (nTags > 0) {
        tags = new ArrayList<>();
        for (int t = 0; t < nTags; t++) {
          tags.add(TAGS[r.nextInt(TAGS.length)].getBytes(StandardCharsets.UTF_8));
        }
      }
      byte[] blob = null;
      if (r.nextInt(5) != 0) {
        blob = new byte[r.nextInt(4)];
        for (int b = 0; b < blob.length; b++) {
          blob[b] = BLOB_BYTES[r.nextInt(BLOB_BYTES.length)];
        }
      }
      rows.add(new Row("d" + i, name, tags, blob, i % 5));
    }
    return rows;
  }

  static Document doc(Row row) {
    Document d = new Document();
    d.add(new StringField("id", row.id, Field.Store.YES));
    if (row.name != null) {
      d.add(new SortedDocValuesField("name", new BytesRef(row.name)));
    }
    if (row.tags != null) {
      for (byte[] t : row.tags) {
        d.add(new SortedSetDocValuesField("tags", new BytesRef(t)));
      }
    }
    if (row.blob != null) {
      d.add(new BinaryDocValuesField("blob", new BytesRef(row.blob)));
    }
    d.add(new NumericDocValuesField("seq", row.seq));
    return d;
  }

  static boolean deleted(int i) {
    return i % 7 == 3;
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("string_sorted_index");
    if (Files.exists(out)) {
      deleteRecursive(out);
    }
    Files.createDirectories(out);

    List<Row> rows = rows();
    StringBuilder m = new StringBuilder();
    m.append("batches=").append(BATCHES).append('\n');
    m.append("per_batch=").append(PER_BATCH).append('\n');
    m.append("num_docs=").append(rows.size()).append('\n');
    for (int i = 0; i < rows.size(); i++) {
      Row row = rows.get(i);
      m.append("doc.").append(i).append('=')
          .append(row.id).append('|')
          .append(row.name == null ? "-" : hex(row.name)).append('|');
      if (row.tags == null) {
        m.append('-');
      } else {
        List<String> hs = new ArrayList<>();
        for (byte[] t : row.tags) {
          hs.add(hex(t));
        }
        m.append(String.join(",", hs));
      }
      m.append('|').append(row.blob == null ? "-" : hex(row.blob))
          .append('|').append(row.seq).append('\n');
    }
    StringBuilder deletedIds = new StringBuilder();
    for (int i = 0; i < rows.size(); i++) {
      if (deleted(i)) {
        deletedIds.append(deletedIds.length() == 0 ? "" : ",").append(rows.get(i).id);
      }
    }
    m.append("deleted=").append(deletedIds).append('\n');

    List<String> names = new ArrayList<>();
    for (Config config : configs()) {
      names.add(config.name);
      Path path = out.resolve(config.name);
      Files.createDirectories(path);
      try (Directory dir = FSDirectory.open(path)) {
        IndexWriterConfig cfg = new IndexWriterConfig();
        cfg.setUseCompoundFile(false);
        cfg.setIndexSort(config.sort);
        cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
        cfg.setRAMBufferSizeMB(256);
        LogDocMergePolicy mp = new LogDocMergePolicy();
        mp.setMergeFactor(1000);
        mp.setNoCFSRatio(0.0);
        cfg.setMergePolicy(mp);
        try (IndexWriter w = new IndexWriter(dir, cfg)) {
          for (int b = 0; b < BATCHES; b++) {
            for (int i = b * PER_BATCH; i < (b + 1) * PER_BATCH; i++) {
              w.addDocument(doc(rows.get(i)));
            }
            w.commit();
          }
          try (DirectoryReader reader = DirectoryReader.open(dir)) {
            if (reader.leaves().size() != BATCHES) {
              throw new IllegalStateException("expected one segment per batch");
            }
            for (LeafReaderContext ctx : reader.leaves()) {
              m.append(config.name).append(".flushed.").append(ctx.ord).append('=')
                  .append(idOrder(ctx)).append('\n');
            }
          }
          for (int i = 0; i < rows.size(); i++) {
            if (deleted(i)) {
              w.deleteDocuments(new Term("id", rows.get(i).id));
            }
          }
          w.commit();
          w.forceMerge(1);
          w.commit();
        }
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
          if (reader.leaves().size() != 1) {
            throw new IllegalStateException("expected one leaf after forceMerge(1)");
          }
          LeafReaderContext ctx = reader.leaves().get(0);
          m.append(config.name).append(".sort=").append(ctx.reader().getMetaData().sort()).append('\n');
          m.append(config.name).append(".merged=").append(idOrder(ctx)).append('\n');
        }
      }
    }
    m.append("configs=").append(String.join(",", names)).append('\n');
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote string_sorted_index/ fixture directory");
  }

  static String idOrder(LeafReaderContext ctx) throws IOException {
    List<String> ids = new ArrayList<>();
    for (int d = 0; d < ctx.reader().maxDoc(); d++) {
      ids.add(ctx.reader().storedFields().document(d).get("id"));
    }
    return String.join(",", ids);
  }

  static String hex(byte[] b) {
    if (b.length == 0) {
      return "~";
    }
    StringBuilder sb = new StringBuilder(b.length * 2);
    for (byte x : b) {
      sb.append(String.format("%02x", x));
    }
    return sb.toString();
  }

  static void deleteRecursive(Path p) throws IOException {
    try (var walk = Files.walk(p)) {
      walk.sorted(java.util.Comparator.reverseOrder()).forEach(f -> {
        try {
          Files.delete(f);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    }
  }
}
