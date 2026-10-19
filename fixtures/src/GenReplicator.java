import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.replicator.nrt.CopyState;
import org.apache.lucene.replicator.nrt.FileMetaData;
import org.apache.lucene.replicator.nrt.PrimaryNode;
import org.apache.lucene.search.SearcherFactory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14.e {@code replicator.nrt}: what a primary node hands a replica. Writes two indexes under
 * {@code replicator/} and {@code replicator/nrt.tsv}:
 *
 * <ul>
 *   <li>{@code plain}: three segments (the second not a compound file), a deletion, no merge;
 *   <li>{@code merged}: the same three segments force-merged into one, with the primary's merged
 *       segment warmer running, so the files it pre-copies are recorded.
 * </ul>
 *
 * <pre>
 * P  tag  version  gen  primaryGen         the NRT point's CopyState
 * F  tag  file  length  checksum  header  footer   every file of the point (FileMetaData; hex)
 * M  tag  file                             CopyState.completedMergeFiles
 * W  tag  file  length  checksum  header  footer   what preCopyMergedSegmentFiles was given
 * I  tag  bytes                            CopyState.infosBytes (hex)
 * </pre>
 *
 * <p>Read by {@code crates/lucene-search/tests/replicator_nrt_fixtures.rs}.
 */
public class GenReplicator {

  private static final HexFormat HEX = HexFormat.of();

  /** A primary that records what its merged segment warmer is asked to pre-copy. */
  static class Primary extends PrimaryNode {
    final Map<String, FileMetaData> warmed = new TreeMap<>();

    Primary(IndexWriter w) throws java.io.IOException {
      super(w, 0, 1, -1, new SearcherFactory(), null);
    }

    @Override
    protected void preCopyMergedSegmentFiles(
        SegmentCommitInfo info, Map<String, FileMetaData> files) {
      warmed.putAll(files);
    }

    @Override
    public void commit() throws java.io.IOException {
      writer.commit();
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("replicator");
    Files.createDirectories(out);
    try (PrintWriter tsv =
        new PrintWriter(Files.newBufferedWriter(out.resolve("nrt.tsv"), StandardCharsets.UTF_8))) {
      build(out.resolve("plain"), tsv, "plain", false);
      build(out.resolve("merged"), tsv, "merged", true);
    }
  }

  private static void build(Path dir, PrintWriter tsv, String tag, boolean merge) throws Exception {
    deleteTree(dir);
    Directory d = FSDirectory.open(dir);
    IndexWriter w =
        new IndexWriter(
            d,
            new IndexWriterConfig()
                .setMergePolicy(merge ? new TieredMergePolicy() : NoMergePolicy.INSTANCE));
    Primary primary = new Primary(w);
    int next = 0;
    for (int batch = 0; batch < 3; batch++) {
      w.getConfig().setUseCompoundFile(batch != 1);
      for (int i = 0; i < 5; i++, next++) {
        Document doc = new Document();
        doc.add(new StringField("id", String.format("k%03d", next), Field.Store.YES));
        doc.add(new TextField("body", "alpha beta gamma " + (next % 3) + " doc" + next, Field.Store.NO));
        doc.add(new NumericDocValuesField("n", next * 7L));
        w.addDocument(doc);
      }
      w.commit();
    }
    w.deleteDocuments(new Term("id", "k003"));
    w.commit();
    if (merge) {
      w.getConfig().setUseCompoundFile(true);
      w.forceMerge(1);
      w.commit();
    }
    primary.flushAndRefresh();
    CopyState cs = primary.getCopyState();
    tsv.printf("P\t%s\t%d\t%d\t%d%n", tag, cs.version(), cs.gen(), cs.primaryGen());
    for (Map.Entry<String, FileMetaData> e : new TreeMap<>(cs.files()).entrySet()) {
      meta(tsv, "F", tag, e.getKey(), e.getValue());
    }
    for (String f : new java.util.TreeSet<>(cs.completedMergeFiles())) {
      tsv.printf("M\t%s\t%s%n", tag, f);
    }
    for (Map.Entry<String, FileMetaData> e : primary.warmed.entrySet()) {
      meta(tsv, "W", tag, e.getKey(), e.getValue());
    }
    tsv.printf("I\t%s\t%s%n", tag, HEX.formatHex(cs.infosBytes()));
    primary.releaseCopyState(cs);
    w.commit();
    primary.close();
    // PrimaryNode.close rolled the writer back and closed the directory; the committed
    // files stay. Drop the lock file a closed writer leaves behind.
    Files.deleteIfExists(dir.resolve("write.lock"));
  }

  private static void meta(PrintWriter tsv, String kind, String tag, String file, FileMetaData m) {
    tsv.printf(
        "%s\t%s\t%s\t%d\t%d\t%s\t%s%n",
        kind, tag, file, m.length(), m.checksum(), HEX.formatHex(m.header()), HEX.formatHex(m.footer()));
  }

  private static void deleteTree(Path dir) throws Exception {
    if (!Files.exists(dir)) return;
    List<Path> all = new ArrayList<>();
    try (var s = Files.walk(dir)) {
      s.forEach(all::add);
    }
    for (int i = all.size() - 1; i >= 0; i--) Files.delete(all.get(i));
  }
}
