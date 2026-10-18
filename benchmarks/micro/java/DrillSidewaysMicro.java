import java.io.IOException;
import java.nio.file.Path;
import org.apache.lucene.facet.DrillDownQuery;
import org.apache.lucene.facet.DrillSideways;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.4c's drill-sideways pair (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_drill_sideways.rs}), with the same case names, over the corpus
 * {@link FacetCountsMicro} builds ({@code build <dir>} delegates to it). Each case is one
 * sequential {@code DrillSideways.search(ddq, 10)}: the {@code DrillSidewaysQuery} search and the
 * {@code FastTaxonomyFacetCounts} of the drill-down and of every sideways dim. One case per
 * {@code DrillSidewaysScorer} strategy: {@code drill_sideways_union} (match-all base, {@code
 * Category/c3}), {@code drill_sideways_advance} (match-all base, {@code Author/a5} and {@code
 * Tag/t7}: the second dim far cheaper than the base) and {@code drill_sideways_query_first}
 * ({@code q:rare} base, {@code Category/c3} and {@code Tag/t7}: the base far cheaper than the
 * dims). Units: searches. Each prints a {@code #check} digest of its hits and of {@code
 * getTopChildren(10, dim)} for each dim first.
 */
public final class DrillSidewaysMicro {
  static void digest(FacetCountsMicro.Fnv f, DrillSideways.DrillSidewaysResult r, String... dims)
      throws IOException {
    f.add(r.hits.totalHits.value());
    for (ScoreDoc sd : r.hits.scoreDocs) {
      f.add(sd.doc);
      f.add(Float.floatToIntBits(sd.score));
    }
    for (String d : dims) f.result(r.facets.getTopChildren(10, d));
  }

  static void run(Path dir) throws IOException {
    FacetsConfig config = FacetCountsMicro.config();
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        Directory td = FSDirectory.open(dir.resolve("taxo"));
        DirectoryReader reader = DirectoryReader.open(d);
        DirectoryTaxonomyReader taxo = new DirectoryTaxonomyReader(td)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      DrillSideways ds = new DrillSideways(searcher, config, taxo);
      Query all = new MatchAllDocsQuery();
      Query rare = new TermQuery(new Term("q", "rare"));

      FacetCountsMicro.cases(
          "drill_sideways_union",
          f -> {
            DrillDownQuery q = new DrillDownQuery(config, all);
            q.add("Category", "c3");
            digest(f, ds.search(q, 10), "Category", "Author");
            return 1;
          });
      FacetCountsMicro.cases(
          "drill_sideways_advance",
          f -> {
            DrillDownQuery q = new DrillDownQuery(config, all);
            q.add("Author", "a5");
            q.add("Tag", "t7");
            digest(f, ds.search(q, 10), "Author", "Tag", "Category");
            return 1;
          });
      FacetCountsMicro.cases(
          "drill_sideways_query_first",
          f -> {
            DrillDownQuery q = new DrillDownQuery(config, rare);
            q.add("Category", "c3");
            q.add("Tag", "t7");
            digest(f, ds.search(q, 10), "Category", "Tag", "Author");
            return 1;
          });
    }
  }

  public static void main(String[] args) throws IOException {
    if (args[0].equals("build")) {
      FacetCountsMicro.main(args);
    } else {
      run(Path.of(args[1]));
    }
  }
}
