import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code SegmentCacheable.isCacheable(ctx)} of weights and values sources, recorded from Lucene for
 * crates/lucene-search/tests/segment_cacheable_fixtures.rs over {@code doc_values_updates_index}
 * (written by {@code GenDocValuesUpdates}; opened read-only here), whose {@code val} and {@code
 * tag} fields carry doc-values updates and whose {@code keep} field does not.
 */
public class GenSegmentCacheable {
  public static void main(String[] args) throws IOException {
    Path data = Path.of(args[0]);
    Path out = data.resolve("segment_cacheable");
    Files.createDirectories(out);
    StringBuilder m = new StringBuilder();
    try (Directory dir = FSDirectory.open(data.resolve("doc_values_updates_index"));
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      List<String> names = new ArrayList<>();
      List<Query> queries = new ArrayList<>();
      names.add("exists_val");
      queries.add(new FieldExistsQuery("val"));
      names.add("exists_tag");
      queries.add(new FieldExistsQuery("tag"));
      names.add("exists_keep");
      queries.add(new FieldExistsQuery("keep"));
      names.add("term");
      queries.add(new TermQuery(new Term("id", "7")));
      names.add("bool_filter_val");
      queries.add(
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("id", "7")), BooleanClause.Occur.MUST)
              .add(new FieldExistsQuery("val"), BooleanClause.Occur.FILTER)
              .build());
      names.add("bool_filter_keep");
      queries.add(
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("id", "7")), BooleanClause.Occur.MUST)
              .add(new FieldExistsQuery("keep"), BooleanClause.Occur.FILTER)
              .build());
      BooleanQuery.Builder big = new BooleanQuery.Builder();
      List<Query> ds = new ArrayList<>();
      for (int i = 0; i < 17; i++) {
        big.add(new TermQuery(new Term("id", Integer.toString(i))), BooleanClause.Occur.SHOULD);
        ds.add(new TermQuery(new Term("id", Integer.toString(i))));
      }
      names.add("bool_17");
      queries.add(big.build());
      names.add("dismax_17");
      queries.add(new DisjunctionMaxQuery(ds, 0f));
      names.add("dismax_tag");
      queries.add(
          new DisjunctionMaxQuery(
              List.of(new TermQuery(new Term("id", "1")), new FieldExistsQuery("tag")), 0f));
      names.add("const_val");
      queries.add(new ConstantScoreQuery(new FieldExistsQuery("val")));
      names.add("boost_keep");
      queries.add(new BoostQuery(new FieldExistsQuery("keep"), 2f));
      for (int i = 0; i < names.size(); i++) {
        m.append("q.").append(names.get(i)).append('=');
        StringBuilder b = new StringBuilder();
        for (LeafReaderContext ctx : reader.leaves()) {
          boolean c =
              searcher
                  .createWeight(searcher.rewrite(queries.get(i)), ScoreMode.COMPLETE_NO_SCORES, 1f)
                  .isCacheable(ctx);
          b.append(c ? '1' : '0');
        }
        m.append(b).append('\n');
      }
      String[] fields = {"val", "tag", "keep", "missing"};
      for (String f : fields) {
        StringBuilder b = new StringBuilder();
        for (LeafReaderContext ctx : reader.leaves()) {
          b.append(DoubleValuesSource.fromLongField(f).isCacheable(ctx) ? '1' : '0');
          b.append(LongValuesSource.fromLongField(f).isCacheable(ctx) ? '1' : '0');
          b.append(DoubleValuesSource.fromLongField(f).toLongValuesSource().isCacheable(ctx) ? '1' : '0');
        }
        m.append("vs.").append(f).append('=').append(b).append('\n');
      }
    }
    Files.writeString(out.resolve("cases.txt"), m.toString(), StandardCharsets.UTF_8);
  }
}
