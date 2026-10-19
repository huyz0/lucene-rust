import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.misc.index.BinaryDocValueSelector;
import org.apache.lucene.misc.index.IndexRearranger;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M14 T14.3 `misc`: {@link IndexRearranger} and {@link BinaryDocValueSelector}. Writes the source
 * index {@code misc/rearrange_src} (three segments of binary-doc-value-keyed documents, three
 * deleted), rearranges it into scratch directories, and writes {@code misc/rearrange.tsv}:
 *
 * <pre>
 * R  case  segment  maxDoc  ids     one segment of the output: its size and its live documents' ids
 * E  case  message                  the rearrange failed with this IllegalStateException message
 * </pre>
 *
 * <p>Read by {@code crates/lucene-search/tests/misc_rearrange_fixtures.rs}.
 */
public class GenMiscRearrange {

  static String id(int n) {
    return String.format("k%03d", n);
  }

  static Set<BytesRef> keys(int from, int to, int step) {
    Set<BytesRef> out = new HashSet<>();
    for (int n = from; n < to; n += step) out.add(new BytesRef(id(n)));
    return out;
  }

  static IndexRearranger.DocumentSelector sel(Set<BytesRef> keys) {
    return new BinaryDocValueSelector("key", keys);
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    Files.createDirectories(out);
    Path src = out.resolve("rearrange_src");
    try (Directory dir = FSDirectory.open(src);
        IndexWriter w =
            new IndexWriter(dir, new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE))) {
      int next = 0;
      for (int size : new int[] {7, 6, 5}) {
        for (int i = 0; i < size; i++, next++) {
          Document doc = new Document();
          doc.add(new StringField("id", id(next), Field.Store.YES));
          doc.add(new BinaryDocValuesField("key", new BytesRef(id(next))));
          w.addDocument(doc);
        }
        w.commit();
      }
      for (int d : new int[] {1, 8, 14}) w.deleteDocuments(new Term("id", id(d)));
      w.commit();
    }

    Path scratch = Files.createTempDirectory("gen-misc-rearrange");
    try (PrintWriter tsv =
            new PrintWriter(
                Files.newBufferedWriter(out.resolve("rearrange.tsv"), StandardCharsets.UTF_8));
        Directory input = FSDirectory.open(src)) {
      run(
          scratch,
          tsv,
          input,
          "geometry",
          BinaryDocValueSelector.createLiveSelectorsFromIndex("key", input),
          BinaryDocValueSelector.createDeleteSelectorFromIndex("key", input));
      run(scratch, tsv, input, "even_odd", List.of(sel(keys(0, 18, 2)), sel(keys(1, 18, 2))), null);
      run(
          scratch,
          tsv,
          input,
          "even_odd_deletes",
          List.of(sel(keys(0, 18, 2)), sel(keys(1, 18, 2))),
          BinaryDocValueSelector.createDeleteSelectorFromIndex("key", input));
      run(
          scratch,
          tsv,
          input,
          "three_ways",
          List.of(sel(keys(12, 18, 1)), sel(keys(0, 5, 1)), sel(keys(5, 12, 1))),
          sel(keys(3, 15, 4)));
      run(scratch, tsv, input, "overlap", List.of(sel(keys(0, 10, 1)), sel(keys(5, 13, 1))), null);
      run(
          scratch,
          tsv,
          input,
          "whole_segment_deleted",
          List.of(sel(keys(0, 7, 1)), sel(keys(7, 18, 1))),
          sel(keys(0, 7, 1)));
    } finally {
      deleteTree(scratch);
    }
    // A rearrange that fails leaves its non-daemon "rearranger" threads running.
    System.exit(0);
  }

  static void run(
      Path scratch,
      PrintWriter tsv,
      Directory input,
      String name,
      List<? extends IndexRearranger.DocumentSelector> selectors,
      IndexRearranger.DocumentSelector deletes)
      throws Exception {
    Path outPath = Files.createTempDirectory(scratch, name);
    try (Directory output = FSDirectory.open(outPath)) {
      try {
        new IndexRearranger(input, output, new IndexWriterConfig(), new ArrayList<>(selectors), deletes)
            .execute();
      } catch (IllegalStateException e) {
        tsv.printf("E\t%s\t%s%n", name, e.getMessage().replaceAll("Document selector .* has matched", "Document selector # has matched"));
        return;
      }
      try (IndexReader reader = DirectoryReader.open(output)) {
        int seg = 0;
        for (LeafReaderContext ctx : reader.leaves()) {
          StringBuilder ids = new StringBuilder();
          var live = ctx.reader().getLiveDocs();
          for (int d = 0; d < ctx.reader().maxDoc(); d++) {
            if (live != null && !live.get(d)) continue;
            if (ids.length() > 0) ids.append(',');
            ids.append(ctx.reader().storedFields().document(d).get("id"));
          }
          tsv.printf("R\t%s\t%d\t%d\t%s%n", name, seg++, ctx.reader().maxDoc(), ids);
        }
      }
    }
  }

  static void deleteTree(Path root) throws Exception {
    try (var walk = Files.walk(root)) {
      walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
    }
  }
}
