import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

/**
 * A field indexed with documents only (no frequencies) but with norms, recorded from Lucene for
 * crates/lucene-search/src/exec/tests.rs.
 *
 * <p>Without frequencies {@code Lucene104PostingsReader} gives a term one impacts level up to
 * {@code NO_MORE_DOCS} holding the impact {@code (freq 1, norm 1)}: every document's frequency is
 * 1 and the shortest non-empty field scores highest, so once a full queue's threshold passes that
 * score nothing is left to visit. Most documents of {@code docs_only} hold one token; every seventh holds
 * two and every eleventh three (longer fields, lower scores), so {@code docs_only:a}'s top hits are its
 * one-token documents and {@code docs_only:c}, only ever the second token, never reaches the bound.
 *
 * <p>One segment of 5,000 documents. Per query, the top 10 under three total-hits thresholds
 * (hits as {@code doc:scoreBits}, the total and its relation).
 */
public class GenDocsOnlyNorms {
  static final int DOCS = 5_000;

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("docs_only_norms_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);

    FieldType kwType = new FieldType();
    kwType.setIndexOptions(IndexOptions.DOCS);
    kwType.setTokenized(true);
    kwType.freeze();

    StringBuilder m = new StringBuilder();
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < DOCS; i++) {
          String first = i % 3 == 0 ? "b" : "a";
          String value = i % 11 == 0 ? first + " c d" : i % 7 == 0 ? first + " c" : first;
          Document doc = new Document();
          doc.add(new Field("docs_only", value, kwType));
          w.addDocument(doc);
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != 1) {
        throw new AssertionError("expected one segment, got " + sis.size());
      }
      int run = 0;
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        String[] terms = {"a", "b", "c"};
        float[] boosts = {1f, 2f};
        for (String t : terms) {
          for (float boost : boosts) {
            Query q = new TermQuery(new Term("docs_only", t));
            if (boost != 1f) {
              q = new BoostQuery(q, boost);
            }
            for (int threshold : new int[] {10, 1000, Integer.MAX_VALUE}) {
              TopScoreDocCollector tc = new TopScoreDocCollectorManager(10, null, threshold).newCollector();
              searcher.search(q, tc);
              TopDocs td = tc.topDocs();
              String k = "run." + run;
              m.append(k).append(".term=").append(t).append('\n');
              m.append(k).append(".boost=").append(Float.floatToIntBits(boost)).append('\n');
              m.append(k).append(".threshold=").append(threshold == Integer.MAX_VALUE ? "max" : threshold).append('\n');
              StringBuilder hits = new StringBuilder();
              for (ScoreDoc sd : td.scoreDocs) {
                if (hits.length() > 0) {
                  hits.append(',');
                }
                hits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
              }
              m.append(k).append(".hits=").append(hits).append('\n');
              m.append(k).append(".total=").append(td.totalHits.value()).append('\n');
              m.append(k).append(".relation=")
                  .append(td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte")
                  .append('\n');
              run++;
            }
          }
        }
      }
      m.insert(0, "run_count=" + run + "\n");
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }
}
