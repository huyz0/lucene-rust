import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MultiReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.misc.index.MultiPassIndexSplitter;
import org.apache.lucene.misc.index.PKIndexSplitter;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M14 T14.3 `misc`: {@link PKIndexSplitter} and {@link MultiPassIndexSplitter}. Writes the source
 * indexes {@code misc/split_a} (three segments, some documents deleted) and {@code misc/split_b}
 * (two segments, none deleted), runs the splitters over them into scratch directories, and writes
 * {@code misc/split.tsv}:
 *
 * <pre>
 * R  case  output  segments  ids        the live documents of one output, in order (ids joined by ",")
 * </pre>
 *
 * <p>Read by {@code crates/lucene-search/tests/misc_split_fixtures.rs}.
 */
public class GenMiscSplit {

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    Files.createDirectories(out);
    Path a = out.resolve("split_a");
    Path b = out.resolve("split_b");
    buildSource(a, 0, new int[] {9, 7, 8}, new int[] {2, 11, 12, 20});
    buildSource(b, 100, new int[] {6, 5}, new int[] {});

    Path scratch = Files.createTempDirectory("gen-misc-split");
    try (PrintWriter tsv =
        new PrintWriter(Files.newBufferedWriter(out.resolve("split.tsv"), StandardCharsets.UTF_8))) {
      // PKIndexSplitter over a mid term, and over a range query.
      pk(scratch, tsv, a, "pk_mid", new Term("id", "k010"), null);
      pk(scratch, tsv, a, "pk_mid_low", new Term("id", "k000"), null);
      pk(scratch, tsv, a, "pk_mid_high", new Term("id", "k999"), null);
      pk(
          scratch,
          tsv,
          a,
          "pk_range",
          null,
          new org.apache.lucene.search.TermRangeQuery(
              "id", new BytesRef("k005"), new BytesRef("k015"), true, true));
      // MultiPassIndexSplitter, sequential and round-robin, over one index and over two.
      for (boolean seq : new boolean[] {true, false}) {
        for (int parts : new int[] {2, 3, 5}) {
          multi(scratch, tsv, new Path[] {a}, parts, seq, "multi_a");
          multi(scratch, tsv, new Path[] {b}, parts, seq, "multi_b");
          multi(scratch, tsv, new Path[] {a, b}, parts, seq, "multi_ab");
        }
      }
    } finally {
      deleteTree(scratch);
    }
  }

  /** Segments of the given sizes (ids {@code k%03d} from {@code base}), then deletes. */
  static void buildSource(Path path, int base, int[] sizes, int[] deletes) throws Exception {
    try (Directory dir = FSDirectory.open(path);
        IndexWriter w =
            new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      int next = base;
      for (int size : sizes) {
        for (int i = 0; i < size; i++, next++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.format("k%03d", next), Field.Store.YES));
          doc.add(new StoredField("n", next * 3));
          w.addDocument(doc);
        }
        w.commit();
      }
      for (int d : deletes) w.deleteDocuments(new Term("id", String.format("k%03d", base + d)));
      w.commit();
    }
  }

  static void pk(
      Path scratch,
      PrintWriter tsv,
      Path src,
      String name,
      Term mid,
      org.apache.lucene.search.Query query)
      throws Exception {
    Path d1 = Files.createTempDirectory(scratch, name);
    Path d2 = Files.createTempDirectory(scratch, name);
    try (Directory in = FSDirectory.open(src);
        Directory o1 = FSDirectory.open(d1);
        Directory o2 = FSDirectory.open(d2)) {
      PKIndexSplitter s =
          mid != null ? new PKIndexSplitter(in, o1, o2, mid) : new PKIndexSplitter(in, o1, o2, query);
      s.split();
      row(tsv, name, "first", o1);
      row(tsv, name, "second", o2);
    }
  }

  static void multi(Path scratch, PrintWriter tsv, Path[] srcs, int parts, boolean seq, String base)
      throws Exception {
    String name = base + (seq ? "_seq_" : "_rr_") + parts;
    List<Directory> opened = new ArrayList<>();
    List<IndexReader> readers = new ArrayList<>();
    Directory[] outs = new Directory[parts];
    try {
      for (Path s : srcs) {
        Directory d = FSDirectory.open(s);
        opened.add(d);
        readers.add(DirectoryReader.open(d));
      }
      for (int i = 0; i < parts; i++) {
        outs[i] = FSDirectory.open(Files.createTempDirectory(scratch, name));
      }
      IndexReader input = readers.size() == 1 ? readers.get(0) : new MultiReader(readers.toArray(IndexReader[]::new));
      new MultiPassIndexSplitter().split(input, outs, seq);
      for (int i = 0; i < parts; i++) row(tsv, name, "part" + i, outs[i]);
    } finally {
      for (IndexReader r : readers) r.close();
      for (Directory d : opened) d.close();
      for (Directory d : outs) if (d != null) d.close();
    }
  }

  static void row(PrintWriter tsv, String name, String output, Directory dir) throws Exception {
    try (IndexReader reader = DirectoryReader.open(dir)) {
      StringBuilder ids = new StringBuilder();
      for (LeafReaderContext ctx : reader.leaves()) {
        var live = ctx.reader().getLiveDocs();
        for (int d = 0; d < ctx.reader().maxDoc(); d++) {
          if (live != null && !live.get(d)) continue;
          if (ids.length() > 0) ids.append(',');
          ids.append(ctx.reader().storedFields().document(d).get("id"));
        }
      }
      tsv.printf("R\t%s\t%s\t%d\t%s%n", name, output, reader.leaves().size(), ids);
    }
  }

  static void deleteTree(Path root) throws Exception {
    try (var walk = Files.walk(root)) {
      walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
    }
  }
}
