import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.search.DummyTotalHitCountCollector;

/** Deterministic callback, manager and live-document fixtures from Lucene 10.5.0. */
public class GenTestFrameworkDummyTotalHitCount {
  public static void main(String[] args) throws Exception {
    StringBuilder out = new StringBuilder();
    for (int segments : new int[] {1, 3}) {
      for (boolean delete : new boolean[] {false, true}) {
        try (var dir = new ByteBuffersDirectory();
             var analyzer = new Analyzer() {
               @Override protected TokenStreamComponents createComponents(String field) {
                 return new TokenStreamComponents(new Tokenizer() {
                   @Override public boolean incrementToken() { return false; }
                 });
               }
             }) {
          var config = new IndexWriterConfig(analyzer).setMergePolicy(NoMergePolicy.INSTANCE);
          try (var writer = new IndexWriter(dir, config)) {
            for (int segment = 0; segment < segments; segment++) {
              for (int id = segment; id < 9; id += segments) {
                var doc = new Document();
                doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
                doc.add(new StringField("tag", id % 2 == 0 ? "even" : "odd", Field.Store.NO));
                writer.addDocument(doc);
              }
              writer.commit();
            }
            if (delete) writer.deleteDocuments(new Term("id", "2"), new Term("id", "7"));
          }
          try (var reader = DirectoryReader.open(dir)) {
            var searcher = new IndexSearcher(reader);
            Query[] queries = {new MatchAllDocsQuery(), new TermQuery(new Term("tag", "even")),
                new TermQuery(new Term("tag", "absent"))};
            String[] names = {"all", "even", "absent"};
            for (int i = 0; i < queries.length; i++) {
              var collector = new DummyTotalHitCountCollector();
              searcher.search(queries[i], collector);
              int managed = searcher.search(queries[i], DummyTotalHitCountCollector.createManager());
              out.append("search\t").append(segments).append('\t').append(delete)
                  .append('\t').append(names[i]).append('\t').append(collector.getTotalHits())
                  .append('\t').append(managed).append('\n');
            }
          }
        }
      }
    }
    var collector = new DummyTotalHitCountCollector();
    var leaf = collector.getLeafCollector(null);
    for (int doc : new int[] {7, 7, 0, 2, 3, 4}) leaf.collect(doc);
    out.append("callbacks\t").append(collector.getTotalHits()).append('\n');
    var count = DummyTotalHitCountCollector.class.getDeclaredField("totalHits");
    count.setAccessible(true);
    count.setInt(collector, Integer.MAX_VALUE);
    leaf.collect(0);
    out.append("overflow\t").append(collector.getTotalHits()).append('\n');
    var other = new DummyTotalHitCountCollector();
    count.setInt(other, -1);
    var manager = DummyTotalHitCountCollector.createManager();
    out.append("reduce\t").append(manager.reduce(List.of(collector, other))).append('\n');
    out.append("empty\t").append(manager.reduce(new ArrayList<>())).append('\n');
    Files.createDirectories(Path.of(args[0]));
    Files.writeString(Path.of(args[0]).resolve("dummy_total_hit_count.txt"), out.toString());
  }
}
