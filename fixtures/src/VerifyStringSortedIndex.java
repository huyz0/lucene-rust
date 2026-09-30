import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Sort;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Stream;

/**
 * Verifies the Rust-written byte-keyed index sorts
 * ({@code crates/lucene-index/examples/write_string_sorted_segment_fixture.rs})
 * against real Lucene's own {@code IndexWriter}.
 *
 * <p>For each configuration directory: every document is read back out of the
 * Rust {@code flushed} index (id and all four doc-values columns) and
 * re-indexed by Lucene in the same batches under the sort Lucene reads out of
 * the Rust {@code .si}. Lucene's segments must hold the documents in exactly
 * the Rust segments' order. Then the documents missing from the Rust
 * {@code merged} index are deleted and Lucene force-merges, and the merged
 * order must match too. Both Rust indexes must pass {@code CheckIndex}, whose
 * {@code testSort} rebuilds every tier's comparator from the {@code .si}.
 */
public class VerifyStringSortedIndex {
  record Row(int n, String id, BytesRef name, List<BytesRef> tags, BytesRef blob, long seq) {}

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    List<Path> configs;
    try (Stream<Path> s = Files.list(root)) {
      configs = s.filter(Files::isDirectory).sorted().toList();
    }
    if (configs.isEmpty()) {
      throw new AssertionError("no configurations under " + root);
    }
    for (Path config : configs) {
      verify(config);
      System.out.println("ok " + config.getFileName());
    }
  }

  static void verify(Path config) throws Exception {
    String name = config.getFileName().toString();
    List<List<String>> rustFlushed = new ArrayList<>();
    List<List<Row>> batches = new ArrayList<>();
    Sort sort;
    try (Directory dir = FSDirectory.open(config.resolve("flushed"));
        DirectoryReader reader = DirectoryReader.open(dir)) {
      sort = reader.leaves().get(0).reader().getMetaData().sort();
      if (sort == null) {
        throw new AssertionError(name + ": the Rust segment declares no sort");
      }
      for (LeafReaderContext ctx : reader.leaves()) {
        if (!sort.equals(ctx.reader().getMetaData().sort())) {
          throw new AssertionError(name + ": segments disagree about the sort");
        }
        List<Row> rows = rows(ctx.reader());
        List<String> order = new ArrayList<>();
        for (Row r : rows) {
          order.add(r.id);
        }
        rustFlushed.add(order);
        List<Row> batch = new ArrayList<>(rows);
        batch.sort(Comparator.comparingInt(Row::n));
        batches.add(batch);
      }
      checkIndex(dir, name + "/flushed");
    }

    List<String> rustMerged;
    Set<String> live = new HashSet<>();
    try (Directory dir = FSDirectory.open(config.resolve("merged"));
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) {
        throw new AssertionError(name + ": merged index has " + reader.leaves().size() + " leaves");
      }
      LeafReader leaf = reader.leaves().get(0).reader();
      if (!sort.equals(leaf.getMetaData().sort())) {
        throw new AssertionError(name + ": merged segment declares " + leaf.getMetaData().sort());
      }
      rustMerged = new ArrayList<>();
      for (Row r : rows(leaf)) {
        rustMerged.add(r.id);
        live.add(r.id);
      }
      checkIndex(dir, name + "/merged");
    }

    try (Directory dir = new ByteBuffersDirectory()) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setIndexSort(sort);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      LogDocMergePolicy mp = new LogDocMergePolicy();
      mp.setMergeFactor(1000);
      mp.setNoCFSRatio(0.0);
      cfg.setMergePolicy(mp);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (List<Row> batch : batches) {
          for (Row r : batch) {
            w.addDocument(doc(r));
          }
          w.commit();
        }
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
          if (reader.leaves().size() != rustFlushed.size()) {
            throw new AssertionError(name + ": Lucene flushed " + reader.leaves().size() + " segments");
          }
          for (LeafReaderContext ctx : reader.leaves()) {
            List<String> javaOrder = new ArrayList<>();
            for (Row r : rows(ctx.reader())) {
              javaOrder.add(r.id);
            }
            if (!javaOrder.equals(rustFlushed.get(ctx.ord))) {
              throw new AssertionError(
                  name + ": flushed segment " + ctx.ord + " differs from Lucene's order\n  rust="
                      + rustFlushed.get(ctx.ord) + "\n  java=" + javaOrder);
            }
          }
        }
        for (List<Row> batch : batches) {
          for (Row r : batch) {
            if (!live.contains(r.id)) {
              w.deleteDocuments(new Term("id", r.id));
            }
          }
        }
        w.commit();
        w.forceMerge(1);
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        List<String> javaOrder = new ArrayList<>();
        for (Row r : rows(reader.leaves().get(0).reader())) {
          javaOrder.add(r.id);
        }
        if (!javaOrder.equals(rustMerged)) {
          throw new AssertionError(
              name + ": merged segment differs from Lucene's order\n  rust=" + rustMerged
                  + "\n  java=" + javaOrder);
        }
      }
    }
  }

  static Document doc(Row r) {
    Document d = new Document();
    d.add(new StringField("id", r.id, Field.Store.YES));
    if (r.name != null) {
      d.add(new SortedDocValuesField("name", r.name));
    }
    if (r.tags != null) {
      for (BytesRef t : r.tags) {
        d.add(new SortedSetDocValuesField("tags", t));
      }
    }
    if (r.blob != null) {
      d.add(new BinaryDocValuesField("blob", r.blob));
    }
    d.add(new NumericDocValuesField("seq", r.seq));
    return d;
  }

  /** Every live document of the leaf, in doc-id order. */
  static List<Row> rows(LeafReader leaf) throws Exception {
    int maxDoc = leaf.maxDoc();
    SortedDocValues names = leaf.getSortedDocValues("name");
    SortedSetDocValues tags = leaf.getSortedSetDocValues("tags");
    BinaryDocValues blobs = leaf.getBinaryDocValues("blob");
    NumericDocValues seqs = leaf.getNumericDocValues("seq");
    List<Row> rows = new ArrayList<>();
    for (int d = 0; d < maxDoc; d++) {
      BytesRef name = null;
      if (names != null && names.advanceExact(d)) {
        name = BytesRef.deepCopyOf(names.lookupOrd(names.ordValue()));
      }
      List<BytesRef> ts = null;
      if (tags != null && tags.advanceExact(d)) {
        ts = new ArrayList<>();
        for (int i = 0; i < tags.docValueCount(); i++) {
          ts.add(BytesRef.deepCopyOf(tags.lookupOrd(tags.nextOrd())));
        }
      }
      BytesRef blob = null;
      if (blobs != null && blobs.advanceExact(d)) {
        blob = BytesRef.deepCopyOf(blobs.binaryValue());
      }
      if (!seqs.advanceExact(d)) {
        throw new AssertionError("doc " + d + " has no seq");
      }
      long seq = seqs.longValue();
      if (leaf.getLiveDocs() != null && !leaf.getLiveDocs().get(d)) {
        continue;
      }
      String id = leaf.storedFields().document(d).get("id");
      rows.add(new Row(Integer.parseInt(id.substring(1)), id, name, ts, blob, seq));
    }
    return rows;
  }

  static void checkIndex(Directory dir, String what) throws Exception {
    try (CheckIndex checker = new CheckIndex(dir)) {
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      ByteArrayOutputStream log = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(log, true, "UTF-8"));
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        System.out.println(log.toString("UTF-8"));
        throw new AssertionError(what + ": CheckIndex is not clean");
      }
    }
  }
}
