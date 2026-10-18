import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.lucene.facet.taxonomy.FacetLabel;
import org.apache.lucene.facet.taxonomy.ParallelTaxonomyArrays;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Verifies {@code crates/lucene-search/examples/write_taxonomy_fixture.rs}: for each of {@code
 * small-cache}, {@code default-cache} and {@code replaced}, CheckIndex is clean, Lucene's {@code
 * DirectoryTaxonomyReader} reads exactly what this port's reader read ({@code dump.tsv}: size,
 * commit user data, every ordinal's parent, youngest child, older sibling and path), every path
 * maps back to its ordinal, and Lucene's {@code DirectoryTaxonomyWriter} appends to the index:
 * every existing category keeps its ordinal, a new one gets the next, the epoch is kept, and the
 * result passes CheckIndex again.
 */
public class VerifyTaxonomy {
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    for (String name : new String[] {"small-cache", "default-cache", "replaced"}) {
      verify(out.resolve(name));
    }
    System.out.println("VerifyTaxonomy: ok");
  }

  static void check(Directory dir, Path path) throws Exception {
    ByteArrayOutputStream log = new ByteArrayOutputStream();
    try (CheckIndex ci = new CheckIndex(dir)) {
      ci.setInfoStream(new PrintStream(log, true, StandardCharsets.UTF_8));
      CheckIndex.Status status = ci.checkIndex();
      if (!status.clean) {
        throw new AssertionError(path + ": CheckIndex failed\n" + log.toString(StandardCharsets.UTF_8));
      }
    }
  }

  static String dump(DirectoryTaxonomyReader r) throws Exception {
    StringBuilder sb = new StringBuilder();
    sb.append("size\t").append(r.getSize()).append('\n');
    Map<String, String> user = new TreeMap<>(r.getCommitUserData());
    for (Map.Entry<String, String> e : user.entrySet()) {
      sb.append("user\t").append(e.getKey()).append('\t').append(e.getValue()).append('\n');
    }
    ParallelTaxonomyArrays arrays = r.getParallelTaxonomyArrays();
    for (int o = 0; o < r.getSize(); o++) {
      sb.append("ord\t").append(o).append('\t').append(arrays.parents().get(o)).append('\t')
          .append(arrays.children().get(o)).append('\t').append(arrays.siblings().get(o));
      FacetLabel label = r.getPath(o);
      for (int i = 0; i < label.length; i++) sb.append('\t').append(label.components[i]);
      sb.append('\n');
    }
    return sb.toString();
  }

  static void verify(Path path) throws Exception {
    String expected = Files.readString(path.resolve("dump.tsv"), StandardCharsets.UTF_8);
    List<FacetLabel> labels = new ArrayList<>();
    String epoch;
    try (Directory dir = FSDirectory.open(path)) {
      check(dir, path);
      try (DirectoryTaxonomyReader r = new DirectoryTaxonomyReader(dir)) {
        String got = dump(r);
        if (!got.equals(expected)) {
          throw new AssertionError(path + ": Lucene reads\n" + got + "\nRust read\n" + expected);
        }
        for (int o = 0; o < r.getSize(); o++) {
          FacetLabel label = r.getPath(o);
          labels.add(label);
          if (r.getOrdinal(label) != o) {
            throw new AssertionError(path + ": ordinal of " + label + " is " + r.getOrdinal(label));
          }
        }
        epoch = r.getCommitUserData().get("index.epoch");
        if (epoch == null) throw new AssertionError(path + ": no index.epoch");
      }
      int size = labels.size();
      try (DirectoryTaxonomyWriter w = new DirectoryTaxonomyWriter(dir, OpenMode.APPEND)) {
        if (w.getSize() != size) throw new AssertionError(path + ": writer size " + w.getSize());
        for (int o = size - 1; o >= 0; o--) {
          int got = w.addCategory(labels.get(o));
          if (got != o) throw new AssertionError(path + ": " + labels.get(o) + " -> " + got + ", not " + o);
        }
        int added = w.addCategory(new FacetLabel("VerifyTaxonomy", "new"));
        if (added != size + 1) throw new AssertionError(path + ": new category got " + added);
      }
      check(dir, path);
      try (DirectoryTaxonomyReader r = new DirectoryTaxonomyReader(dir)) {
        if (r.getSize() != size + 2) throw new AssertionError(path + ": size after append " + r.getSize());
        if (!epoch.equals(r.getCommitUserData().get("index.epoch"))) {
          throw new AssertionError(path + ": epoch changed by an append");
        }
      }
    }
  }
}
