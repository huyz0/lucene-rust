import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Stream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.CommonTermsQuery;
import org.apache.lucene.queries.mlt.MoreLikeThis;
import org.apache.lucene.queries.mlt.MoreLikeThisQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M10 T10.6's more-like-this and common-terms fixture: {@code mlt/index}, a four-segment index
 * with deletions in two segments, and two result files.
 *
 * <p>Fields: {@code id} (StringField), {@code body} (stored text, no term vectors), {@code tv}
 * (stored text with term vectors), {@code title} (stored text). The words follow a skewed
 * distribution, so some are in most documents and some in a few.
 *
 * <p>{@code mlt.tsv}: {@code MoreLikeThis} under twelve settings -- for six documents (one deleted)
 * the interesting terms, the query's clauses (field, term, boost bits) and its hits; {@code
 * like(field, texts)} and {@code like(Map)}; and {@code MoreLikeThisQuery} over three texts, with
 * hits and explanations. {@code common.tsv}: {@code CommonTermsQuery} over fourteen term sets and
 * settings, hits with score bits and four explanations each; then eight boosted booleans of
 * MUST/FILTER/SHOULD/MUST_NOT term, phrase and nested-boolean clauses (two under {@code
 * ClassicSimilarity}), hits and four explanations each. Exceptions are recorded by class name.
 *
 * <p>Usage: {@code java GenMoreLikeThis <fixtures-data-dir>}.
 */
public class GenMoreLikeThis {
  static final String[] WORDS = {
    "the", "river", "stone", "light", "house", "garden", "winter", "summer", "bridge", "forest",
    "market", "window", "letter", "station", "harbor", "orchard", "meadow", "lantern", "compass",
    "anchor", "violin", "glacier", "tundra", "quartz", "zephyr"
  };

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
      }
    }
  }

  static final FieldType TV = new FieldType(TextField.TYPE_STORED);

  static {
    TV.setStoreTermVectors(true);
    TV.freeze();
  }

  static String text(Random r, int n) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) b.append(' ');
      int k = (int) (Math.pow(r.nextDouble(), 2.2) * WORDS.length);
      String w = WORDS[Math.min(k, WORDS.length - 1)];
      b.append(r.nextInt(13) == 0 ? w.toUpperCase(java.util.Locale.ROOT) : w);
    }
    return b.toString();
  }

  static Document doc(Random r, int id) {
    Document d = new Document();
    d.add(new StringField("id", "d" + id, Field.Store.NO));
    if (id % 9 != 4) d.add(new TextField("body", text(r, 3 + r.nextInt(30)), Field.Store.YES));
    if (id % 7 != 2) d.add(new Field("tv", text(r, 2 + r.nextInt(25)), TV));
    if (id % 3 == 0) d.add(new TextField("title", text(r, 1 + r.nextInt(5)), Field.Store.YES));
    if (id % 5 == 1) d.add(new TextField("body", text(r, 4), Field.Store.YES));
    return d;
  }

  static IndexWriterConfig config() {
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setRAMBufferSizeMB(256);
    return cfg;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("mlt");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(0x10_5_2026_1067L);
    try (Directory dir = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(dir, config())) {
      int id = 0;
      int[] sizes = {40, 30, 35, 15};
      for (int seg = 0; seg < sizes.length; seg++) {
        for (int i = 0; i < sizes[seg]; i++, id++) {
          w.addDocument(doc(r, id));
        }
        w.commit();
        if (seg == 1 || seg == 2) {
          for (int k = 0; k < 6; k++) {
            w.deleteDocuments(new Term("id", "d" + (id - 1 - r.nextInt(sizes[seg]))));
          }
          w.commit();
        }
      }
    }
    StringBuilder mlt = new StringBuilder();
    StringBuilder common = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 4) throw new AssertionError("segments");
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      mlt(mlt, reader, searcher);
      common(common, searcher);
      boosted(common, searcher);
    }
    Files.writeString(root.resolve("mlt.tsv"), mlt.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("common.tsv"), common.toString(), StandardCharsets.UTF_8);
  }

  static String hex(float f) {
    return Integer.toHexString(Float.floatToIntBits(f));
  }

  static String err(Throwable e) {
    return "!" + e.getClass().getSimpleName();
  }

  static String clean(String s) {
    return s == null ? "null" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  interface Get {
    String get() throws Exception;
  }

  static String g(Get f) {
    try {
      return clean(f.get());
    } catch (Exception e) {
      return err(e);
    }
  }

  static String hits(IndexSearcher s, Query q) throws IOException {
    TopDocs td = s.search(q, 1000);
    StringBuilder b = new StringBuilder();
    b.append(td.totalHits.value()).append(' ');
    for (ScoreDoc sd : td.scoreDocs) b.append(sd.doc).append(':').append(hex(sd.score)).append(',');
    return b.toString();
  }

  /** The clauses: occur, field:term, and the boost's bits when boosted. */
  static String clauses(BooleanQuery bq) {
    StringBuilder b = new StringBuilder();
    b.append("msm=").append(bq.getMinimumNumberShouldMatch());
    for (BooleanClause c : bq.clauses()) {
      b.append(' ').append(c.occur().name().charAt(0));
      Query q = c.query();
      String boost = "";
      if (q instanceof BoostQuery bq2) {
        boost = "^" + hex(bq2.getBoost());
        q = bq2.getQuery();
      }
      Term t = ((TermQuery) q).getTerm();
      b.append(t.field()).append(':').append(t.text()).append(boost);
    }
    return b.toString();
  }

  /** {name, fields, minTermFreq, minDocFreq, maxDocFreq (-1 default), maxQueryTerms, boost,
   * boostFactor, minWordLen, maxWordLen, stop words (| separated or -)}. */
  static final String[][] SETTINGS = {
    {"defaults", "body", "2", "5", "-1", "25", "false", "1", "0", "0", "-"},
    {"tf1", "body", "1", "1", "-1", "25", "false", "1", "0", "0", "-"},
    {"tv", "tv", "1", "1", "-1", "25", "false", "1", "0", "0", "-"},
    {"both", "body|tv", "1", "2", "-1", "8", "false", "1", "0", "0", "-"},
    {"boosted", "body|tv|title", "1", "1", "-1", "6", "true", "2.5", "0", "0", "-"},
    {"few", "body", "1", "1", "-1", "3", "true", "1", "0", "0", "-"},
    {"words", "tv", "1", "1", "-1", "25", "false", "1", "5", "6", "-"},
    {"stops", "body", "1", "1", "-1", "25", "false", "1", "0", "0", "the|river|stone"},
    {"maxdf", "body", "1", "1", "40", "25", "false", "1", "0", "0", "-"},
    {"maxdfpct", "tv", "1", "1", "pct30", "25", "false", "1", "0", "0", "-"},
    {"nofields", "title|missing", "1", "1", "-1", "25", "false", "1", "0", "0", "-"},
    {"noanalyzer", "body", "1", "1", "-1", "25", "false", "1", "0", "0", "-"},
  };

  static final int[] DOCS = {0, 3, 11, 17, 44, 102};

  static MoreLikeThis configure(DirectoryReader reader, String[] s) throws IOException {
    MoreLikeThis m = new MoreLikeThis(reader);
    m.setFieldNames(s[1].split("\\|"));
    m.setMinTermFreq(Integer.parseInt(s[2]));
    m.setMinDocFreq(Integer.parseInt(s[3]));
    if (s[4].startsWith("pct")) m.setMaxDocFreqPct(Integer.parseInt(s[4].substring(3)));
    else if (!s[4].equals("-1")) m.setMaxDocFreq(Integer.parseInt(s[4]));
    m.setMaxQueryTerms(Integer.parseInt(s[5]));
    m.setBoost(Boolean.parseBoolean(s[6]));
    m.setBoostFactor(Float.parseFloat(s[7]));
    m.setMinWordLen(Integer.parseInt(s[8]));
    m.setMaxWordLen(Integer.parseInt(s[9]));
    if (!s[10].equals("-")) m.setStopWords(Set.of(s[10].split("\\|")));
    if (!s[0].equals("noanalyzer")) m.setAnalyzer(new StandardAnalyzer());
    return m;
  }

  static final String[] TEXTS = {
    "river stone light house river garden stone winter",
    "glacier tundra quartz QUARTZ zephyr anchor anchor",
    "the the the bridge",
  };

  static void mlt(StringBuilder out, DirectoryReader reader, IndexSearcher searcher)
      throws IOException {
    for (String[] s : SETTINGS) {
      for (int doc : DOCS) {
        String head = s[0] + "\tdoc " + doc;
        out.append(head).append("\tterms\t").append(g(() -> {
          String[] t = configure(reader, s).retrieveInterestingTerms(doc);
          return String.join(",", t);
        })).append('\n');
        out.append(head).append("\tquery\t").append(g(() ->
            clauses((BooleanQuery) configure(reader, s).like(doc)))).append('\n');
        out.append(head).append("\thits\t").append(g(() ->
            hits(searcher, configure(reader, s).like(doc)))).append('\n');
      }
      for (int i = 0; i < TEXTS.length; i++) {
        final String text = TEXTS[i];
        final String next = TEXTS[(i + 1) % TEXTS.length];
        String head = s[0] + "\ttext " + i;
        out.append(head).append("\tterms\t").append(g(() ->
            String.join(",", configure(reader, s).retrieveInterestingTerms(new StringReader(text), "body"))))
            .append('\n');
        out.append(head).append("\tquery\t").append(g(() ->
            clauses((BooleanQuery) configure(reader, s).like("body", new StringReader(text), new StringReader(next)))))
            .append('\n');
        out.append(head).append("\tmap\t").append(g(() -> {
          Map<String, Collection<Object>> m = new HashMap<>();
          m.put("body", List.of(text));
          m.put("tv", List.of(next, 42));
          return clauses((BooleanQuery) configure(reader, s).like(m));
        })).append('\n');
      }
    }
    // MoreLikeThisQuery
    String[][] mltq = {
      {"0", "body", "body", "0.3", "1", "5", "-1", "-"},
      {"1", "body|tv", "tv", "0.5", "1", "4", "2", "-"},
      {"0", "tv", "body", "0", "2", "10", "-1", "river"},
      {"2", "body", "body", "1", "1", "5", "-1", "-"},
    };
    for (String[] q : mltq) {
      String head = "mltq\t" + String.join(",", q);
      MoreLikeThisQuery mq =
          new MoreLikeThisQuery(TEXTS[Integer.parseInt(q[0])], q[1].split("\\|"), new StandardAnalyzer(), q[2]);
      mq.setPercentTermsToMatch(Float.parseFloat(q[3]));
      mq.setMinTermFrequency(Integer.parseInt(q[4]));
      mq.setMaxQueryTerms(Integer.parseInt(q[5]));
      mq.setMinDocFreq(Integer.parseInt(q[6]));
      if (!q[7].equals("-")) mq.setStopWords(Set.of(q[7].split("\\|")));
      out.append(head).append("\trewrite\t").append(g(() ->
          clauses((BooleanQuery) searcher.rewrite(mq)))).append('\n');
      out.append(head).append("\thits\t").append(g(() -> hits(searcher, mq))).append('\n');
      for (int doc : new int[] {0, 5, 50}) {
        out.append(head).append("\texplain ").append(doc).append('\t')
            .append(g(() -> searcher.explain(mq, doc).toString())).append('\n');
      }
    }
  }

  /** {terms (| separated, field:text), maxTermFrequency, high occur, low occur, low msm, high msm,
   * low boost, high boost}. */
  static final String[][] COMMON = {
    {"body:river|body:glacier|body:quartz", "0.3", "SHOULD", "SHOULD", "0", "0", "1", "1"},
    {"body:river|body:stone|body:glacier|body:tundra", "0.25", "SHOULD", "MUST", "0", "0", "1", "1"},
    {"body:river|body:stone|body:light", "0.1", "SHOULD", "SHOULD", "0", "0", "1", "1"},
    {"body:river|body:stone|body:light", "0.1", "MUST", "SHOULD", "0", "0", "1", "1"},
    {"body:river|body:stone|body:light", "0.1", "SHOULD", "SHOULD", "0", "2", "1", "1"},
    {"body:river|body:glacier|body:zephyr|body:quartz|body:anchor", "20", "SHOULD", "SHOULD", "0.5", "0", "1.5", "0.5"},
    {"body:river|body:glacier|body:zephyr|body:quartz|body:anchor", "20", "MUST", "SHOULD", "2", "0", "1", "2"},
    {"body:glacier|body:nosuch|tv:river", "0.05", "SHOULD", "SHOULD", "0", "0", "1", "1"},
    {"body:river", "0.1", "SHOULD", "SHOULD", "0", "0", "1", "1"},
    {"", "0.1", "SHOULD", "SHOULD", "0", "0", "1", "1"},
    {"title:river|title:stone|title:quartz", "0.1", "SHOULD", "SHOULD", "0.4", "0.6", "1", "1"},
    {"body:river|body:stone", "0.1", "MUST_NOT", "SHOULD", "0", "0", "1", "1"},
    {"body:river|body:glacier|body:quartz", "0.3", "SHOULD", "FILTER", "0", "0", "2", "1"},
    {"body:river|body:stone|body:glacier|body:tundra", "0.25", "MUST", "FILTER", "0", "0", "1", "0.5"},
  };

  /**
   * A boosted boolean: {clauses (space separated, +/#/- prefixed, in the MUST, FILTER, SHOULD,
   * MUST_NOT order the Rust query keeps them in), boost, and optionally the searcher's similarity
   * ("classic")}. A clause is a term {@code f:t}, a phrase {@code "f:a,b"} or a nested boolean of
   * SHOULD terms {@code (f:a|f:b)}. The filters and exclusions explain through weights created
   * without scores: a term's and a phrase's dummy scorers, a nested boolean's sum of them.
   */
  static final String[][] BOOSTED = {
    {"+body:river #body:stone -body:glacier", "2.5"},
    {"#body:light body:river body:stone -body:quartz", "0.5"},
    {"#body:river #body:stone", "3"},
    {"+body:stone -body:river -body:zephyr", "1.5"},
    {"+body:river #\"body:the,the\" -body:quartz", "2"},
    {"#(body:light|body:winter) body:river -\"body:light,house\"", "1.5"},
    {"+body:river #\"body:the,the\" #(body:light|body:winter) -body:quartz", "1", "classic"},
    {"#body:river -\"body:river,window\" -(body:glacier|body:compass)", "1", "classic"},
  };

  static Query clause(String t) {
    if (t.startsWith("\"")) {
      String[] ft = t.substring(1, t.length() - 1).split(":");
      return new PhraseQuery(ft[0], ft[1].split(","));
    }
    if (t.startsWith("(")) {
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      for (String c : t.substring(1, t.length() - 1).split("\\|")) {
        b.add(clause(c), BooleanClause.Occur.SHOULD);
      }
      return b.build();
    }
    String[] ft = t.split(":");
    return new TermQuery(new Term(ft[0], ft[1]));
  }

  static void boosted(StringBuilder out, IndexSearcher plain) {
    for (String[] spec : BOOSTED) {
      IndexSearcher searcher = plain;
      String sim = "";
      if (spec.length > 2) {
        searcher = new IndexSearcher(plain.getIndexReader());
        searcher.setQueryCache(null);
        searcher.setSimilarity(new ClassicSimilarity());
        sim = "@" + spec[2];
      }
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      for (String c : spec[0].split(" ")) {
        BooleanClause.Occur occur;
        String t = c;
        switch (c.charAt(0)) {
          case '+' -> { occur = BooleanClause.Occur.MUST; t = c.substring(1); }
          case '#' -> { occur = BooleanClause.Occur.FILTER; t = c.substring(1); }
          case '-' -> { occur = BooleanClause.Occur.MUST_NOT; t = c.substring(1); }
          default -> occur = BooleanClause.Occur.SHOULD;
        }
        b.add(clause(t), occur);
      }
      Query q = new BoostQuery(b.build(), Float.parseFloat(spec[1]));
      String head = "boost\t" + spec[0] + "^" + spec[1] + sim;
      IndexSearcher s = searcher;
      out.append(head).append("\thits\t").append(g(() -> hits(s, q))).append('\n');
      for (int doc : new int[] {0, 7, 33, 60}) {
        out.append(head).append("\texplain ").append(doc).append('\t')
            .append(g(() -> s.explain(q, doc).toString())).append('\n');
      }
    }
  }

  static void common(StringBuilder out, IndexSearcher searcher) {
    for (String[] c : COMMON) {
      String head = String.join(",", c);
      CommonTermsQuery q;
      try {
        q = new CommonTermsQuery(
            BooleanClause.Occur.valueOf(c[2]), BooleanClause.Occur.valueOf(c[3]), Float.parseFloat(c[1]));
      } catch (Exception e) {
        out.append(head).append("\tnew\t").append(err(e)).append('\n');
        continue;
      }
      if (!c[0].isEmpty()) {
        for (String t : c[0].split("\\|")) {
          String[] ft = t.split(":");
          q.add(new Term(ft[0], ft[1]));
        }
      }
      q.setLowFreqMinimumNumberShouldMatch(Float.parseFloat(c[4]));
      q.setHighFreqMinimumNumberShouldMatch(Float.parseFloat(c[5]));
      // The boosts have no setters: subclass-only fields.
      final float lowBoost = Float.parseFloat(c[6]);
      final float highBoost = Float.parseFloat(c[7]);
      CommonTermsQuery boosted =
          new CommonTermsQuery(q.getHighFreqOccur(), q.getLowFreqOccur(), q.getMaxTermFrequency()) {
            {
              this.lowFreqBoost = lowBoost;
              this.highFreqBoost = highBoost;
            }
          };
      for (Term t : q.getTerms()) boosted.add(t);
      boosted.setLowFreqMinimumNumberShouldMatch(q.getLowFreqMinimumNumberShouldMatch());
      boosted.setHighFreqMinimumNumberShouldMatch(q.getHighFreqMinimumNumberShouldMatch());
      out.append(head).append("\ttostring\t").append(clean(boosted.toString())).append('\n');
      out.append(head).append("\thits\t").append(g(() -> hits(searcher, boosted))).append('\n');
      for (int doc : new int[] {0, 7, 33, 60}) {
        out.append(head).append("\texplain ").append(doc).append('\t')
            .append(g(() -> searcher.explain(boosted, doc).toString())).append('\n');
      }
    }
  }
}
