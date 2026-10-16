import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.misc.index.BPIndexReorderer;
import org.apache.lucene.misc.index.BPReorderingMergePolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14 T14.3 `misc`: {@link BPReorderingMergePolicy}. Writes {@code misc/bp_merge_src} (three
 * segments of 150 documents, every seventh deleted) and {@code misc/bp_merge.tsv}:
 *
 * <pre>
 * O  config  position  id        the ids of the merged segment in order, after forceMerge(1)
 * G  config  position  reordered the merged segment's bp.reordered diagnostic
 * </pre>
 *
 * <p>Configurations: {@code forced} (every merge reordered) and {@code too_small}
 * (minNaturalMergeNumDocs above the merge's size, so the natural merge is left in order, run as a
 * non-forced merge through {@code maybeMerge}).
 *
 * <p>Read by {@code crates/lucene-search/tests/misc_bp_fixtures.rs}.
 */
public class GenMiscBpMerge {

  static String body(Random r) {
    StringBuilder sb = new StringBuilder();
    int topic = r.nextInt(12);
    int terms = 4 + r.nextInt(5);
    for (int t = 0; t < terms; t++) {
      int tp = r.nextInt(5) == 0 ? (topic + 1) % 12 : topic;
      sb.append("w").append(tp).append('_').append(r.nextInt(9)).append(' ');
    }
    return sb.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    Files.createDirectories(out);
    Path src = out.resolve("bp_merge_src");
    Random r = new Random(7);
    try (Directory dir = FSDirectory.open(src);
        IndexWriter w =
            new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (int seg = 0; seg < 15; seg++) {
        for (int i = 0; i < 30; i++) {
          int n = seg * 30 + i;
          Document doc = new Document();
          doc.add(new StringField("id", "d" + n, Field.Store.YES));
          doc.add(new TextField("body", body(r), Field.Store.NO));
          w.addDocument(doc);
        }
        w.commit();
      }
      for (int n = 0; n < 450; n += 7) {
        w.deleteDocuments(new Term("id", "d" + n));
      }
      w.commit();
    }

    // computeDocMap over the segments' merged view, in leaf order
    BPIndexReorderer view = new BPIndexReorderer();
    view.setRAMBudgetMB(256);
    view.setMinDocFreq(2);
    view.setMinPartitionSize(4);
    view.setFields(Set.of("body"));
    StringBuilder map = new StringBuilder();
    try (Directory dir = FSDirectory.open(src);
        IndexReader reader = DirectoryReader.open(dir);
        Directory tmp = new org.apache.lucene.store.ByteBuffersDirectory()) {
      java.util.List<org.apache.lucene.index.CodecReader> leaves = new java.util.ArrayList<>();
      for (LeafReaderContext ctx : reader.leaves()) {
        leaves.add((org.apache.lucene.index.CodecReader) ctx.reader());
      }
      // package-private in Lucene: reached by reflection (same class path, no modules)
      java.lang.reflect.Method wrap =
          Class.forName("org.apache.lucene.index.SlowCompositeCodecReaderWrapper")
              .getDeclaredMethod("wrap", java.util.List.class);
      wrap.setAccessible(true);
      org.apache.lucene.index.CodecReader merged =
          (org.apache.lucene.index.CodecReader) wrap.invoke(null, leaves);
      org.apache.lucene.index.Sorter.DocMap dm = view.computeDocMap(merged, tmp, null);
      for (int i = 0; i < dm.size(); i++) {
        if (i > 0) map.append(' ');
        map.append(dm.newToOld(i));
      }
    }

    Path scratch = Files.createTempDirectory("gen-misc-bp-merge");
    try (PrintWriter tsv =
        new PrintWriter(Files.newBufferedWriter(out.resolve("bp_merge.tsv"), StandardCharsets.UTF_8))) {
      tsv.printf("V\tview\t%s%n", map);
      run(tsv, "identity", src, scratch.resolve("identity"), 1, true);
      run(tsv, "forced", src, scratch.resolve("forced"), 1, true);
      run(tsv, "natural", src, scratch.resolve("natural"), 1, false);
      run(tsv, "too_small", src, scratch.resolve("too_small"), Integer.MAX_VALUE, false);
    } finally {
      try (var walk = Files.walk(scratch)) {
        walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    System.exit(0);
  }

  static void run(
      PrintWriter tsv, String config, Path src, Path copy, int minNaturalNumDocs, boolean forced)
      throws Exception {
    Files.createDirectories(copy);
    try (var list = Files.list(src)) {
      for (Path p : (Iterable<Path>) list::iterator) {
        if (!p.getFileName().toString().equals("write.lock")) {
          Files.copy(p, copy.resolve(p.getFileName()));
        }
      }
    }
    BPIndexReorderer reorderer = new BPIndexReorderer();
    reorderer.setRAMBudgetMB(256);
    reorderer.setMinDocFreq(2);
    reorderer.setMinPartitionSize(4);
    reorderer.setFields(Set.of(config.equals("identity") ? "none" : "body"));
    TieredMergePolicy tmp = new TieredMergePolicy();
    tmp.setNoCFSRatio(0);
    tmp.setSegmentsPerTier(2);
    tmp.setFloorSegmentMB(0.001);
    BPReorderingMergePolicy policy = new BPReorderingMergePolicy(tmp, reorderer);
    policy.setMinNaturalMergeNumDocs(minNaturalNumDocs);
    try (Directory dir = FSDirectory.open(copy);
        IndexWriter w = new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(policy).setMergeScheduler(new org.apache.lucene.index.SerialMergeScheduler()))) {
      if (forced) {
        w.forceMerge(1);
      } else {
        // natural merges
        w.maybeMerge();
      }
      w.commit();
    }
    try (Directory dir = FSDirectory.open(copy);
        IndexReader reader = DirectoryReader.open(dir)) {
      int pos = 0;
      for (LeafReaderContext ctx : reader.leaves()) {
        SegmentReader sr = (SegmentReader) ctx.reader();
        for (int d = 0; d < ctx.reader().maxDoc(); d++) {
          if (sr.getLiveDocs() != null && !sr.getLiveDocs().get(d)) continue;
          tsv.printf("O\t%s\t%d\t%s%n", config, pos++, sr.storedFields().document(d).get("id"));
        }
      }
      int seg = 0;
      for (LeafReaderContext ctx : reader.leaves()) {
        SegmentReader sr = (SegmentReader) ctx.reader();
        tsv.printf(
            "G\t%s\t%d\t%s%n",
            config, seg++, sr.getSegmentInfo().info.getDiagnostics().get("bp.reordered"));
      }
    }
  }
}
