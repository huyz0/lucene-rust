import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.misc.index.IndexSplitter;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14 T14.3 `misc`: {@link IndexSplitter}. Writes {@code misc/index_splitter_src} (four segments,
 * one with a deletion) and {@code misc/index_splitter.tsv}:
 *
 * <pre>
 * L  segment  size                       listSegments() of the source, one line per segment
 * S  position  segment  maxDoc  ids      split(dest, [_2, _0]): the destination's segments in order
 * C  counter                             the destination's segment-name counter
 * M  minVersion                          the destination's minimum segment version
 * R  position  segment  maxDoc  ids      remove([_1, _3, none]) on a copy of the source: what is left
 * F  file                                files in the copy after the remove that the source also has
 * </pre>
 *
 * <p>Read by {@code crates/lucene-search/tests/misc_index_splitter_fixtures.rs}.
 */
public class GenMiscIndexSplitter {

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    Files.createDirectories(out);
    Path src = out.resolve("index_splitter_src");
    try (Directory dir = FSDirectory.open(src);
        IndexWriter w =
            new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      int next = 0;
      for (int size : new int[] {4, 3, 5, 2}) {
        for (int i = 0; i < size; i++, next++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.format("k%03d", next), Field.Store.YES));
          w.addDocument(doc);
        }
        w.commit();
      }
      w.deleteDocuments(new Term("id", "k005"));
      w.commit();
    }

    Path scratch = Files.createTempDirectory("gen-misc-index-splitter");
    try (PrintWriter tsv =
        new PrintWriter(
            Files.newBufferedWriter(out.resolve("index_splitter.tsv"), StandardCharsets.UTF_8))) {
      // listSegments prints to System.out; recompute the same lines from the splitter's infos.
      IndexSplitter lister = new IndexSplitter(src);
      java.io.PrintStream old = System.out;
      java.io.ByteArrayOutputStream captured = new java.io.ByteArrayOutputStream();
      System.setOut(new java.io.PrintStream(captured, true, StandardCharsets.UTF_8));
      lister.listSegments();
      System.setOut(old);
      for (String line : captured.toString(StandardCharsets.UTF_8).split("\\R")) {
        if (line.isEmpty()) continue;
        int sp = line.indexOf(' ');
        tsv.printf("L\t%s\t%s%n", line.substring(0, sp), line.substring(sp + 1));
      }

      Path dest = scratch.resolve("dest");
      new IndexSplitter(src).split(dest, new String[] {"_2", "_0"});
      try (Directory d = FSDirectory.open(dest)) {
        dump(tsv, "S", d);
        SegmentInfos sis = SegmentInfos.readLatestCommit(d);
        tsv.printf("C\t%d%n", sis.counter);
        tsv.printf("M\t%s%n", sis.getMinSegmentLuceneVersion());
      }

      Path copy = scratch.resolve("copy");
      Files.createDirectories(copy);
      try (var list = Files.list(src)) {
        for (Path p : (Iterable<Path>) list::iterator) {
          if (!p.getFileName().toString().equals("write.lock")) {
            Files.copy(p, copy.resolve(p.getFileName()));
          }
        }
      }
      new IndexSplitter(copy).remove(new String[] {"_1", "_3", "none"});
      try (Directory d = FSDirectory.open(copy)) {
        dump(tsv, "R", d);
        for (String f : d.listAll()) {
          if (Files.exists(src.resolve(f))) tsv.printf("F\t%s%n", f);
        }
      }
    } finally {
      try (var walk = Files.walk(scratch)) {
        walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
  }

  static void dump(PrintWriter tsv, String tag, Directory dir) throws Exception {
    try (IndexReader reader = DirectoryReader.open(dir)) {
      int pos = 0;
      for (LeafReaderContext ctx : reader.leaves()) {
        StringBuilder ids = new StringBuilder();
        var live = ctx.reader().getLiveDocs();
        for (int d = 0; d < ctx.reader().maxDoc(); d++) {
          if (live != null && !live.get(d)) continue;
          if (ids.length() > 0) ids.append(',');
          ids.append(ctx.reader().storedFields().document(d).get("id"));
        }
        String name = ((org.apache.lucene.index.SegmentReader) ctx.reader()).getSegmentName();
        tsv.printf("%s\t%d\t%s\t%d\t%s%n", tag, pos++, name, ctx.reader().maxDoc(), ids);
      }
    }
  }
}
