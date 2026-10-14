import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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
import org.apache.lucene.queries.spans.SpanNotQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.complexPhrase.ComplexPhraseQueryParser;
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
 * M13 T13.1: {@code ComplexPhraseQueryParser} over {@code
 * fixtures/corpus/queryparser-complexphrase.txt}, field {@code body}, in order and not, against
 * an index of twelve documents in two segments ({@code index/}). A row of {@code <config>.tsv} is the escaped query,
 * then {@code Q}, {@code toString("body")}, the class tree, the {@code IndexSearcher.rewrite} of
 * the query (canonical: a boolean's clauses sorted within each occur) or
 * {@code X} and its exception, and the matching documents; or {@code E}/{@code X} and the parse's
 * exception. {@code crates/lucene-search/tests/queryparser_complexphrase_fixtures.rs} compares.
 */
public class GenQueryParserComplexPhrase {

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("queryparser_complexphrase");
    Path indexDir = out.resolve("index");
    Files.createDirectories(indexDir);
    FieldType type = new FieldType();
    type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    type.setTokenized(true);
    type.freeze();
    try (FSDirectory dir = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()).setOpenMode(IndexWriterConfig.OpenMode.CREATE).setMergePolicy(NoMergePolicy.INSTANCE))) {
      for (int i = 0; i < GenQueryParserSurround.BODIES.length; i++) {
        Document d = new Document();
        d.add(new Field("body", GenQueryParserSurround.BODIES[i], type));
        d.add(new Field("title", GenQueryParserSurround.TITLES[i], type));
        w.addDocument(d);
      }
      w.commit();
      // a second segment: 100 terms two edits from "aaxy" (fuzzy ties past
      // maxExpansions) and 1100 "zz" terms (past the clause limit)
      StringBuilder aa = new StringBuilder();
      for (int i = 0; i < 100; i++) aa.append(String.format("aa%02d ", i));
      StringBuilder zz = new StringBuilder();
      for (int i = 0; i < 1100; i++) zz.append(String.format("zz%04d ", i));
      for (String body : new String[] {aa.toString(), zz.toString()}) {
        Document d = new Document();
        d.add(new Field("body", body, type));
        w.addDocument(d);
      }
      w.commit();
    }
    List<String> corpus = AnalysisRows.corpus("queryparser-complexphrase.txt");
    try (FSDirectory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      for (boolean inOrder : new boolean[] {true, false}) {
        StringBuilder b = new StringBuilder();
        for (String q : corpus) {
          b.append(AnalysisRows.esc(q)).append('\t');
          ComplexPhraseQueryParser p = new ComplexPhraseQueryParser("body", new StandardAnalyzer());
          p.setInOrder(inOrder);
          try {
            Query parsed = p.parse(q);
            b.append("Q\t").append(AnalysisRows.esc(parsed.toString("body")))
                .append('\t').append(AnalysisRows.esc(GenQueryParserClassic.describe(parsed))).append('\t');
            try {
              Query rewritten = searcher.rewrite(parsed);
              b.append(AnalysisRows.esc(canon(rewritten))).append('\t').append(GenQueryParserSurround.hits(searcher, rewritten));
            } catch (RuntimeException e) {
              b.append("X ").append(AnalysisRows.esc(e.getClass().getSimpleName() + ": " + e.getMessage())).append('\t');
            }
          } catch (ParseException e) {
            b.append("E\t").append(AnalysisRows.esc(e.getMessage()));
          } catch (RuntimeException e) {
            b.append("X\t").append(AnalysisRows.esc(e.getClass().getSimpleName() + ": " + e.getMessage()));
          }
          b.append('\n');
        }
        Files.writeString(out.resolve(inOrder ? "in_order.tsv" : "unordered.tsv"), b.toString(), StandardCharsets.UTF_8);
      }
    }
  }

  /** As {@link GenQueryParserSurround#canon}, with span-nots, a span-or's clauses in their order, a boolean's sorted. */
  static String canon(Query q) {
    if (q instanceof BooleanQuery bq) {
      StringBuilder b = new StringBuilder("Bool(");
      for (BooleanClause.Occur occur : new BooleanClause.Occur[] {
        BooleanClause.Occur.MUST, BooleanClause.Occur.FILTER, BooleanClause.Occur.SHOULD, BooleanClause.Occur.MUST_NOT
      }) {
        List<String> parts = new ArrayList<>();
        for (BooleanClause c : bq.clauses()) {
          if (c.occur() == occur) {
            parts.add(switch (occur) {
              case MUST -> "+";
              case MUST_NOT -> "-";
              case FILTER -> "#";
              case SHOULD -> "";
            } + canon(c.query()));
          }
        }
        Collections.sort(parts);
        for (String part : parts) b.append(part).append(' ');
      }
      b.append(')');
      if (bq.getMinimumNumberShouldMatch() > 0) b.append('~').append(bq.getMinimumNumberShouldMatch());
      return b.toString();
    }
    if (q instanceof BoostQuery bq) return "Boost(" + canon(bq.getQuery()) + "^" + bq.getBoost() + ")";
    if (q instanceof SpanOrQuery so) {
      List<String> parts = new ArrayList<>();
      for (SpanQuery c : so.getClauses()) parts.add(canon(c));
      return "SpanOr(" + String.join(",", parts) + ")";
    }
    if (q instanceof SpanNearQuery sn) {
      List<String> parts = new ArrayList<>();
      for (SpanQuery c : sn.getClauses()) parts.add(canon(c));
      return "SpanNear([" + String.join(",", parts) + "]," + sn.getSlop() + "," + sn.isInOrder() + ")";
    }
    if (q instanceof SpanNotQuery sn) {
      return "SpanNot(" + canon(sn.getInclude()) + "," + canon(sn.getExclude()) + ")";
    }
    if (q instanceof SpanTermQuery st) return "SpanTerm(" + st.getTerm() + ")";
    if (q instanceof TermQuery t) return "Term(" + t.getTerm() + ")";
    if (q instanceof MatchNoDocsQuery) return "None";
    return q.getClass().getSimpleName() + "(" + q + ")";
  }
}
