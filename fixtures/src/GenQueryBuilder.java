import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.QueryBuilder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.Stream;

/**
 * {@code QueryBuilder} over {@code StandardAnalyzer} text and over canned token graphs (a
 * {@code CannedTokenStream}-like stream of {@code term:posInc:posLen} tokens: stacked synonyms,
 * multi-token synonyms spanning positions, holes), recorded for
 * crates/lucene-search/tests/query_builder_fixtures.rs.
 *
 * <p>{@code query_builder_index}: one segment of 300 documents whose {@code body} mixes a small
 * vocabulary (the canned terms among it). {@code cases.tsv}: {@code method TAB input TAB toString
 * TAB hits TAB total}, the query's {@code toString("body")} ({@code null} when the builder returns
 * none), the top 10 of {@code IndexSearcher.search(query, 10)} as {@code doc:scoreBitsHex} and
 * the total hit count. Methods: {@code B} {@code createBooleanQuery} (SHOULD), {@code BM} (MUST),
 * {@code P<slop>} {@code createPhraseQuery}, {@code M<fraction>} {@code createMinShouldMatchQuery};
 * a leading {@code C} runs the same over a canned stream through {@code createFieldQuery(TokenStream,
 * ...)}, and a trailing {@code A} sets {@code autoGenerateMultiTermSynonymsPhraseQuery}, {@code G}
 * clears {@code enableGraphQueries}, {@code I} clears {@code enablePositionIncrements}.
 */
public class GenQueryBuilder {
  static final String[] WORDS = {
    "fast", "wi", "fi", "wifi", "network", "new", "york", "ny", "city", "quick", "brown", "fox",
    "dog", "canine", "the", "lazy", "jumps", "over", "red", "blue",
  };

  static final String[][] TEXT_CASES = {
    {"B", ""},
    {"B", "Fox"},
    {"B", "quick brown fox"},
    {"BM", "quick brown fox"},
    {"B", "The Quick-Brown fox's"},
    {"P0", "quick brown"},
    {"P0", "brown fox jumps"},
    {"P2", "quick fox"},
    {"M0.5", "quick brown fox dog"},
    {"M0.34", "red blue lazy"},
    {"M1.0", "quick fox"},
    {"M0.5", "fox"},
  };

  static final String[][] CANNED_CASES = {
    {"CB", "fast:1:1,wi:1:1,wifi:0:2,fi:1:1,network:1:1"},
    {"CBM", "fast:1:1,wi:1:1,wifi:0:2,fi:1:1,network:1:1"},
    {"CBA", "fast:1:1,wi:1:1,wifi:0:2,fi:1:1,network:1:1"},
    {"CP0", "fast:1:1,wi:1:1,wifi:0:2,fi:1:1,network:1:1"},
    {"CP1", "fast:1:1,wi:1:1,wifi:0:2,fi:1:1,network:1:1"},
    {"CBG", "fast:1:1,wi:1:1,wifi:0:2,fi:1:1,network:1:1"},
    {"CB", "new:1:1,ny:0:2,york:1:1,city:1:1"},
    {"CBM", "fast:1:1,wi:1:1,wifi:0:2,fi:1:1,new:1:1,ny:0:2,york:1:1"},
    {"CP0", "new:1:1,ny:0:2,york:1:1,city:1:1"},
    {"CB", "quick:1:1,fast:0:1,fox:1:1"},
    {"CBM", "quick:1:1,fast:0:1,fox:1:1"},
    {"CB", "dog:1:1,canine:0:1"},
    {"CP0", "quick:1:1,fast:0:1,brown:1:1"},
    {"CP0I", "quick:1:1,fast:0:1,brown:2:1"},
    {"CP0", "quick:1:1,brown:2:1"},
    {"CP0I", "quick:1:1,brown:2:1"},
    {"CM0.5", "quick:1:1,fast:0:1,fox:1:1,dog:1:1,canine:0:1"},
  };

  /** Tokens {@code term:posInc:posLen}, offsets by token index. */
  static final class Canned extends TokenStream {
    final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    final PositionIncrementAttribute inc = addAttribute(PositionIncrementAttribute.class);
    final PositionLengthAttribute len = addAttribute(PositionLengthAttribute.class);
    final OffsetAttribute off = addAttribute(OffsetAttribute.class);
    final String[][] tokens;
    int upto;

    Canned(String spec) {
      String[] parts = spec.split(",");
      tokens = new String[parts.length][];
      for (int i = 0; i < parts.length; i++) tokens[i] = parts[i].split(":");
    }

    @Override
    public boolean incrementToken() {
      if (upto == tokens.length) return false;
      clearAttributes();
      String[] t = tokens[upto];
      term.setEmpty().append(t[0]);
      inc.setPositionIncrement(Integer.parseInt(t[1]));
      len.setPositionLength(Integer.parseInt(t[2]));
      off.setOffset(upto, upto + 1);
      upto++;
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      upto = 0;
    }
  }

  static final class Exposed extends QueryBuilder {
    Exposed(Analyzer a) {
      super(a);
    }

    Query fromStream(TokenStream ts, BooleanClause.Occur op, boolean quoted, int slop) {
      return createFieldQuery(ts, op, "body", quoted, slop);
    }
  }

  static Query build(Exposed b, String method, String input) {
    boolean canned = method.startsWith("C");
    String m = canned ? method.substring(1) : method;
    if (m.endsWith("A")) {
      b.setAutoGenerateMultiTermSynonymsPhraseQuery(true);
      m = m.substring(0, m.length() - 1);
    }
    if (m.endsWith("G")) {
      b.setEnableGraphQueries(false);
      m = m.substring(0, m.length() - 1);
    }
    if (m.endsWith("I")) {
      b.setEnablePositionIncrements(false);
      m = m.substring(0, m.length() - 1);
    }
    if (m.equals("B") || m.equals("BM")) {
      BooleanClause.Occur op = m.equals("B") ? BooleanClause.Occur.SHOULD : BooleanClause.Occur.MUST;
      return canned
          ? b.fromStream(new Canned(input), op, false, 0)
          : b.createBooleanQuery("body", input, op);
    }
    if (m.startsWith("P")) {
      int slop = Integer.parseInt(m.substring(1));
      return canned
          ? b.fromStream(new Canned(input), BooleanClause.Occur.MUST, true, slop)
          : b.createPhraseQuery("body", input, slop);
    }
    if (m.startsWith("M")) {
      float fraction = Float.parseFloat(m.substring(1));
      if (!canned) return b.createMinShouldMatchQuery("body", input, fraction);
      // createMinShouldMatchQuery over a stream: the SHOULD boolean, then the minimum.
      Query q = b.fromStream(new Canned(input), BooleanClause.Occur.SHOULD, false, 0);
      if (q instanceof org.apache.lucene.search.BooleanQuery bq) {
        org.apache.lucene.search.BooleanQuery.Builder mb =
            new org.apache.lucene.search.BooleanQuery.Builder();
        mb.setMinimumNumberShouldMatch((int) (fraction * bq.clauses().size()));
        for (BooleanClause c : bq) mb.add(c);
        q = mb.build();
      }
      return q;
    }
    throw new IllegalArgumentException(method);
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("query_builder_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Path index = out.resolve("index");
    Files.createDirectories(index);
    Random r = new Random(20260930L);
    StringBuilder cases = new StringBuilder();
    try (Directory dir = FSDirectory.open(index)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int id = 0; id < 300; id++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
          StringBuilder body = new StringBuilder();
          for (int i = 0, n = 2 + r.nextInt(12); i < n; i++) {
            if (i > 0) body.append(' ');
            body.append(WORDS[r.nextInt(WORDS.length)]);
          }
          doc.add(new TextField("body", body.toString(), Field.Store.NO));
          w.addDocument(doc);
        }
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        StandardAnalyzer analyzer = new StandardAnalyzer();
        for (String[][] group : new String[][][] {TEXT_CASES, CANNED_CASES}) {
          for (String[] c : group) {
            Query q = build(new Exposed(analyzer), c[0], c[1]);
            cases.append(c[0]).append('\t').append(c[1]).append('\t');
            if (q == null) {
              cases.append("null\t-\t0\n");
              continue;
            }
            cases.append(q.toString("body")).append('\t');
            TopDocs td = searcher.search(q, 10);
            StringBuilder hits = new StringBuilder();
            for (ScoreDoc sd : td.scoreDocs) {
              if (hits.length() > 0) hits.append(',');
              hits.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(sd.score)));
            }
            cases.append(hits.length() == 0 ? "-" : hits).append('\t');
            cases.append(td.totalHits.value()).append('\n');
          }
        }
      }
    }
    Files.writeString(out.resolve("cases.tsv"), cases.toString(), StandardCharsets.UTF_8);
  }
}
