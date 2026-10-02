import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M9 T9.5's write-path proof: real Lucene opens the index this port's {@code IndexWriter} wrote
 * through its spatial-extras strategies ({@code write_spatial_strategies_fixture}, from {@code
 * GenSpatialStrategies}' documents), finds it clean ({@link CheckIndex}), and answers every
 * question of {@code fixtures/data/spatial_strategies/queries.tsv} -- every strategy x operation
 * x query shape, the value sources over every document, heatmaps, date facets -- exactly as it
 * answers over its own index of the same documents ({@code index/} beside it). Both are answered
 * here, in one JVM, by {@link SpatialExtrasCorpus}, so the comparison holds whatever the JVM's
 * trig intrinsics do (the recorded answers were made with them off).
 *
 * <p>Usage: {@code java VerifySpatialExtras <index-dir> <fixtures/data/spatial_strategies>}.
 */
public class VerifySpatialExtras {
  public static void main(String[] args) throws Exception {
    Path rust = Path.of(args[0]);
    Path fixture = Path.of(args[1]);
    try (Directory rd = FSDirectory.open(rust);
        CheckIndex ci = new CheckIndex(rd)) {
      CheckIndex.Status st = ci.checkIndex();
      if (!st.clean) throw new AssertionError("CheckIndex failed on the Rust index");
    }
    int n = 0;
    try (Directory rd = FSDirectory.open(rust);
        DirectoryReader rr = DirectoryReader.open(rd);
        Directory jd = FSDirectory.open(fixture.resolve("index"));
        DirectoryReader jr = DirectoryReader.open(jd)) {
      if (rr.leaves().size() != jr.leaves().size()) {
        throw new AssertionError(rr.leaves().size() + " segments, Lucene's has " + jr.leaves().size());
      }
      IndexSearcher rs = new IndexSearcher(rr);
      IndexSearcher js = new IndexSearcher(jr);
      rs.setQueryCache(null);
      js.setQueryCache(null);
      for (String line : Files.readAllLines(fixture.resolve("queries.tsv"))) {
        String question = line.substring(0, line.indexOf("\t=>\t"));
        String[] a = question.split("\t");
        String got = SpatialExtrasCorpus.answer(rs, a);
        String want = SpatialExtrasCorpus.answer(js, a);
        if (!got.equals(want)) {
          throw new AssertionError(
              "differs: " + question + "\n  lucene's index: " + want + "\n  rust's index:   " + got);
        }
        n++;
      }
    }
    if (n < 1200) throw new AssertionError("only " + n + " questions");
    System.out.println(
        "VerifySpatialExtras: CheckIndex clean, " + n + " questions answered as over Lucene's index");
  }
}
