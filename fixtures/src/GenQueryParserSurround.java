import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queryparser.surround.parser.ParseException;
import org.apache.lucene.queryparser.surround.parser.QueryParser;
import org.apache.lucene.queryparser.surround.query.BasicQueryFactory;
import org.apache.lucene.queryparser.surround.query.SrndQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.1: the surround query language over {@code fixtures/corpus/queryparser-surround.txt}, an
 * index of ten documents ({@code index/}, fields {@code body} and {@code title}) and, per query,
 * {@code surround.tsv}: the escaped query, then {@code P}, {@link SrndQuery#toString()}, the
 * rewritten Lucene query for field {@code body} (canonical: a {@code SpanOrQuery}'s clauses
 * sorted, as a {@code HashMap}'s order is unspecified), the matching document ids, the rewrite
 * under {@code BasicQueryFactory(3)}; or {@code E} and the {@code ParseException}'s message, or
 * {@code X} and another exception. {@code crates/lucene-search/tests/queryparser_surround_fixtures.rs}
 * compares.
 */
public class GenQueryParserSurround {

  static final String[] BODIES = {
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog outpaces a quick fox",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
    "the five boxing wizards jump quickly",
    "jackdaws love my big sphinx of quartz",
    "the lazy dog sleeps while the quick fox runs",
    "foxes and dogs and boxes and jugs",
    "quick quick quick brown brown dog"
  };
  static final String[] TITLES = {
    "fox story", "dog story", "box story", "zebra story", "sphinx story",
    "wizard story", "quartz story", "fox and dog", "jugs", "quick brown"
  };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("queryparser_surround");
    Path indexDir = out.resolve("index");
    Files.createDirectories(indexDir);
    FieldType type = new FieldType();
    type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    type.setTokenized(true);
    type.setStored(false);
    type.freeze();
    try (FSDirectory dir = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()).setOpenMode(IndexWriterConfig.OpenMode.CREATE).setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (int i = 0; i < BODIES.length; i++) {
        Document d = new Document();
        d.add(new Field("body", BODIES[i], type));
        d.add(new Field("title", TITLES[i], type));
        w.addDocument(d);
      }
      w.commit();
    }
    List<String> corpus = AnalysisRows.corpus("queryparser-surround.txt");
    StringBuilder b = new StringBuilder();
    try (FSDirectory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      for (String q : corpus) {
        b.append(AnalysisRows.esc(q)).append('\t');
        try {
          SrndQuery parsed = QueryParser.parse(q);
          b.append("P\t").append(AnalysisRows.esc(parsed.toString()));
          for (int max : new int[] {1024, 3}) {
            b.append('\t');
            try {
              Query lq = parsed.makeLuceneQueryField("body", new BasicQueryFactory(max));
              Query rewritten = searcher.rewrite(lq);
              b.append(AnalysisRows.esc(canon(rewritten)));
              if (max == 1024) {
                b.append('\t').append(hits(searcher, rewritten));
              }
            } catch (Exception e) {
              b.append("X ").append(AnalysisRows.esc(e.getClass().getSimpleName() + ": " + e.getMessage()));
              if (max == 1024) {
                b.append('\t');
              }
            }
          }
        } catch (ParseException e) {
          b.append("E\t").append(AnalysisRows.esc(e.getMessage()));
        } catch (RuntimeException | Error e) {
          b.append("X\t").append(AnalysisRows.esc(e.getClass().getSimpleName() + ": " + e.getMessage()));
        }
        b.append('\n');
      }
    }
    Files.writeString(out.resolve("surround.tsv"), b.toString(), StandardCharsets.UTF_8);
  }

  static String hits(IndexSearcher searcher, Query q) throws Exception {
    ScoreDoc[] docs = searcher.search(q, 100).scoreDocs;
    List<Integer> ids = new ArrayList<>();
    for (ScoreDoc d : docs) ids.add(d.doc);
    Collections.sort(ids);
    return ids.isEmpty() ? "-" : ids.toString().replace(" ", "");
  }

  /** The query as the Rust test writes it: booleans by occur, span-or clauses sorted. */
  static String canon(Query q) {
    if (q instanceof BooleanQuery bq) {
      StringBuilder b = new StringBuilder("Bool(");
      for (BooleanClause.Occur occur : new BooleanClause.Occur[] {
        BooleanClause.Occur.MUST, BooleanClause.Occur.FILTER, BooleanClause.Occur.SHOULD, BooleanClause.Occur.MUST_NOT
      }) {
        for (BooleanClause c : bq.clauses()) {
          if (c.occur() == occur) {
            b.append(switch (occur) {
              case MUST -> "+";
              case MUST_NOT -> "-";
              case FILTER -> "#";
              case SHOULD -> "";
            }).append(canon(c.query())).append(' ');
          }
        }
      }
      b.append(')');
      if (bq.getMinimumNumberShouldMatch() > 0) b.append('~').append(bq.getMinimumNumberShouldMatch());
      return b.toString();
    }
    if (q instanceof BoostQuery bq) {
      return "Boost(" + canon(bq.getQuery()) + "^" + bq.getBoost() + ")";
    }
    if (q instanceof SpanOrQuery so) {
      List<String> parts = new ArrayList<>();
      for (SpanQuery c : so.getClauses()) parts.add(canon(c));
      Collections.sort(parts);
      return "SpanOr(" + String.join(",", parts) + ")";
    }
    if (q instanceof SpanNearQuery sn) {
      List<String> parts = new ArrayList<>();
      for (SpanQuery c : sn.getClauses()) parts.add(canon(c));
      return "SpanNear([" + String.join(",", parts) + "]," + sn.getSlop() + "," + sn.isInOrder() + ")";
    }
    if (q instanceof SpanTermQuery st) return "SpanTerm(" + st.getTerm() + ")";
    if (q instanceof TermQuery t) return "Term(" + t.getTerm() + ")";
    if (q instanceof MatchNoDocsQuery) return "None";
    return q.getClass().getSimpleName() + "(" + q + ")";
  }
}
