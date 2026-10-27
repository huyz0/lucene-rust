import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SoftDeletesDirectoryReaderWrapper;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.Bits;

/**
 * Differential fixture for {@code SoftDeletesDirectoryReaderWrapper}, soft updates and update by
 * query as {@code RandomIndexWriter} uses them (M14.h): for each of {@value #SEEDS} seeds, an
 * {@code IndexWriter} with a soft-deletes field and {@code NoMergePolicy} runs {@value #OPS}
 * operations drawn from {@code new Random(seed)}, and each reader taken on the way is described.
 *
 * <p>Writes {@code soft_deletes.txt}: one line per reader, {@code <seed> <op> dir|nrt} then the
 * reader's leaves as {@code maxDoc/numDocs} and its live ids, sorted. A {@code dir} reader is a
 * {@code SoftDeletesDirectoryReaderWrapper} over {@code DirectoryReader.open(dir)} after a
 * commit; an {@code nrt} reader is {@code DirectoryReader.open(writer)}, which resolves soft
 * deletes itself.
 *
 * <p>Usage: java GenTestFrameworkSoftDeletes <outdir>
 */
public class GenTestFrameworkSoftDeletes {
  static final int SEEDS = 40;
  static final int OPS = 150;
  static final String SOFT = "___soft_deletes";
  static final StringBuilder OUT = new StringBuilder();

  static Document doc(int id) {
    Document d = new Document();
    d.add(new StringField("id", Integer.toString(id), Field.Store.YES));
    return d;
  }

  static Term id(int id) {
    return new Term("id", Integer.toString(id));
  }

  static void describe(long seed, int op, String kind, DirectoryReader reader) throws Exception {
    OUT.append(seed).append(' ').append(op).append(' ').append(kind);
    TreeSet<Integer> live = new TreeSet<>();
    for (LeafReaderContext ctx : reader.leaves()) {
      LeafReader leaf = ctx.reader();
      OUT.append(' ').append(leaf.maxDoc()).append('/').append(leaf.numDocs());
      Bits liveDocs = leaf.getLiveDocs();
      StoredFields stored = leaf.storedFields();
      for (int doc = 0; doc < leaf.maxDoc(); doc++) {
        if (liveDocs == null || liveDocs.get(doc)) {
          live.add(Integer.parseInt(stored.document(doc).get("id")));
        }
      }
    }
    OUT.append(" |");
    for (int id : live) {
      OUT.append(' ').append(id);
    }
    OUT.append('\n');
  }

  static void run(long seed) throws Exception {
    try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
      IndexWriterConfig c = new IndexWriterConfig();
      c.setSoftDeletesField(SOFT);
      c.setMergePolicy(NoMergePolicy.INSTANCE);
      IndexWriter w = new IndexWriter(dir, c);
      Random ops = new Random(seed);
      for (int i = 0; i < OPS; i++) {
        int k = ops.nextInt(30);
        switch (ops.nextInt(12)) {
          case 0, 1, 2, 3 -> w.softUpdateDocument(id(k), doc(k), new NumericDocValuesField(SOFT, 1));
          case 4 -> w.deleteDocuments(id(k));
          case 5 -> w.updateDocument(id(k), doc(k));
          case 6 -> {
            w.commit();
            try (DirectoryReader r = DirectoryReader.open(dir)) {
              describe(seed, i, "dir", new SoftDeletesDirectoryReaderWrapper(r, SOFT));
            }
          }
          case 7 -> {
            try (DirectoryReader r = DirectoryReader.open(w)) {
              describe(seed, i, "nrt", r);
            }
          }
          case 8 -> w.updateDocuments(new TermQuery(id(k)), List.of(doc(k)));
          case 9 -> w.flush();
          default -> w.addDocument(doc(k));
        }
      }
      w.commit();
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        describe(seed, OPS, "dir", new SoftDeletesDirectoryReaderWrapper(r, SOFT));
      }
      w.close();
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    for (long seed = 0; seed < SEEDS; seed++) {
      run(seed);
    }
    Files.writeString(out.resolve("soft_deletes.txt"), OUT.toString(), StandardCharsets.UTF_8);
  }
}
