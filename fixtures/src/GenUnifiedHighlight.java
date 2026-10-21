import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.spans.SpanContainingQuery;
import org.apache.lucene.queries.spans.SpanFirstQuery;
import org.apache.lucene.queries.spans.SpanMultiTermQueryWrapper;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanNotQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queries.spans.SpanWithinQuery;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.search.SynonymQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.search.uhighlight.CustomSeparatorBreakIterator;
import org.apache.lucene.search.uhighlight.DefaultPassageFormatter;
import org.apache.lucene.search.uhighlight.LengthGoalBreakIterator;
import org.apache.lucene.search.uhighlight.Passage;
import org.apache.lucene.search.uhighlight.PassageFormatter;
import org.apache.lucene.search.uhighlight.SplittingBreakIterator;
import org.apache.lucene.search.uhighlight.UnifiedHighlighter;
import org.apache.lucene.search.uhighlight.WholeBreakIterator;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.2 ({@code uhighlight}): an index of {@code matchhighlight-docs.tsv} whose first value
 * column is indexed five ways (postings offsets, positions only, docs only, term vectors,
 * postings offsets plus term vectors) and what {@code UnifiedHighlighter} makes of every query of
 * {@code uhighlight-queries.txt} ({@code F:} stands for the field being highlighted) under six
 * configurations, over every document. {@code uhighlight.tsv}: configuration, query, field,
 * {@code R}/{@code X}, then the snippet of each document ({@code \u0002} for none).
 * {@code breaks.tsv}: random texts through Lucene's break iterators. {@code uhighlight-built.tsv}
 * (M14): hand-built queries (spans, positioned and multi-phrases, synonyms, mixed fields) the
 * query parser cannot write. {@code uhighlight-calls.tsv} (M14): {@code
 * highlightFieldsAsObjects} under three stored-value batching thresholds, with the formatter's
 * call order. {@code
 * crates/lucene-search/tests/uhighlight_fixtures.rs} compares.
 */
public class GenUnifiedHighlight {

  static String esc(String s) {
    return AnalysisRows.esc(s);
  }

  static FieldType type(IndexOptions o, boolean tv) {
    return type(o, tv, true);
  }

  static FieldType type(IndexOptions o, boolean tv, boolean tvOffsets) {
    FieldType t = new FieldType();
    t.setIndexOptions(o);
    t.setTokenized(true);
    t.setStored(true);
    if (tv) {
      t.setStoreTermVectors(true);
      t.setStoreTermVectorPositions(true);
      t.setStoreTermVectorOffsets(tvOffsets);
    }
    t.freeze();
    return t;
  }

  static final String[] FIELDS = {"offs", "pos", "docs", "tv", "both", "tvpos"};

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("uhighlight");
    Files.createDirectories(out);
    Analyzer analyzer = new StandardAnalyzer();

    Path indexDir = out.resolve("index");
    if (Files.exists(indexDir)) {
      try (var walk = Files.walk(indexDir)) {
        walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    FieldType offs = type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS, false);
    FieldType pos = type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS, false);
    FieldType docs = type(IndexOptions.DOCS, false);
    FieldType tv = type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS, true);
    // term vectors with positions but no offsets: highlighting must refuse
    FieldType tvpos = type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS, true, false);
    FieldType both = type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS, true);
    try (FSDirectory fs = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(fs, new IndexWriterConfig(analyzer).setUseCompoundFile(false))) {
      for (String line : AnalysisRows.corpus("matchhighlight-docs.tsv")) {
        String[] c = line.split("\t", -1);
        Document d = new Document();
        d.add(new Field("id", c[0], org.apache.lucene.document.StringField.TYPE_STORED));
        for (String v : c[1].split("\u0001")) {
          d.add(new Field("offs", v, offs));
          d.add(new Field("tv", v, tv));
          d.add(new Field("both", v, both));
          d.add(new Field("tvpos", v, tvpos));
        }
        for (String v : c[2].split("\u0001")) d.add(new Field("pos", v, pos));
        for (String v : c[3].split("\u0001")) d.add(new Field("docs", v, docs));
        w.addDocument(d);
      }
      w.forceMerge(1);
    }

    StringBuilder rows = new StringBuilder();
    try (FSDirectory fs = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(fs)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      QueryParser parser = new QueryParser("offs", analyzer);
      List<String> queries = AnalysisRows.corpus("uhighlight-queries.txt");
      int[] all = new int[reader.maxDoc()];
      for (int i = 0; i < all.length; i++) all[i] = i;
      for (int cfg = 0; cfg < 6; cfg++) {
        for (int qi = 0; qi < queries.size(); qi++) {
          for (String field : FIELDS) {
            String text = queries.get(qi);
            boolean templated = text.contains("F:");
            if (!templated && !field.equals("offs")) continue;
            Query q = parser.parse(text.replace("F:", field + ":"));
            UnifiedHighlighter.Builder b = UnifiedHighlighter.builder(searcher, analyzer);
            int maxPassages = 2;
            switch (cfg) {
              case 0:
                break;
              case 1:
                b.withWeightMatches(false).withHighlightPhrasesStrictly(false);
                break;
              case 2:
                b.withWeightMatches(false)
                    .withHighlightPhrasesStrictly(false)
                    .withPassageRelevancyOverSpeed(false);
                break;
              case 3:
                b.withBreakIterator(
                        () -> LengthGoalBreakIterator.createMinLength(
                            BreakIterator.getSentenceInstance(Locale.ROOT), 50, 0.5f))
                    .withFormatter(new DefaultPassageFormatter("[", "]", " ... ", true))
                    .withMaxNoHighlightPassages(1);
                maxPassages = 3;
                break;
              case 4:
                b.withFieldMatcher(f -> true)
                    .withBreakIterator(() -> new CustomSeparatorBreakIterator(','))
                    .withMaxLength(60);
                break;
              default:
                b.withHandleMultiTermQuery(false)
                    .withBreakIterator(() -> new WholeBreakIterator())
                    .withMaxNoHighlightPassages(0);
                maxPassages = 1;
            }
            UnifiedHighlighter uh = b.build();
            rows.append(cfg).append((char) 9).append(qi).append((char) 9).append(field).append((char) 9);
            try {
              String[] snippets = uh.highlightFields(new String[] {field}, q, all, new int[] {maxPassages}).get(field);
              StringBuilder joined = new StringBuilder();
              for (int i = 0; i < snippets.length; i++) {
                if (i > 0) joined.append((char) 31);
                joined.append(snippets[i] == null ? "\u0002" : snippets[i]);
              }
              rows.append("R").append((char) 9).append(esc(joined.toString()));
            } catch (Exception e) {
              rows.append("X").append((char) 9).append(esc(e.getClass().getName()));
            }
            rows.append((char) 10);
          }
        }
      }
    }
    Files.writeString(out.resolve("uhighlight.tsv"), rows.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("breaks.tsv"), breaks(), StandardCharsets.UTF_8);
    try (FSDirectory fs = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(fs)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      Files.writeString(
          out.resolve("uhighlight-built.tsv"), built(searcher, analyzer), StandardCharsets.UTF_8);
      Files.writeString(
          out.resolve("uhighlight-calls.tsv"), calls(searcher, analyzer), StandardCharsets.UTF_8);
    }
  }

  static Term t(String f, String text) {
    return new Term(f, text);
  }

  static SpanQuery st(String f, String text) {
    return new SpanTermQuery(t(f, text));
  }

  static Query bool(int msm, Object... occurQuery) {
    BooleanQuery.Builder b = new BooleanQuery.Builder().setMinimumNumberShouldMatch(msm);
    for (int i = 0; i < occurQuery.length; i += 2) {
      b.add((Query) occurQuery[i + 1], (BooleanClause.Occur) occurQuery[i]);
    }
    return b.build();
  }

  static final BooleanClause.Occur MUST = BooleanClause.Occur.MUST;
  static final BooleanClause.Occur SHOULD = BooleanClause.Occur.SHOULD;
  static final BooleanClause.Occur NOT = BooleanClause.Occur.MUST_NOT;

  static final int BUILT = 23;

  /**
   * The hand-built queries of {@code uhighlight-built.tsv} over field {@code f}: what the query
   * parser cannot build (spans, multi-phrase, synonyms, phrases with gaps) and clauses on other
   * fields. {@code uhighlight_fixtures.rs} builds the same, by number.
   */
  static Query built(int i, String f) {
    switch (i) {
      case 0:
        return new SpanNearQuery(new SpanQuery[] {st(f, "quick"), st(f, "fox")}, 1, false);
      case 1:
        return new PhraseQuery(2, f, "fox", "quick");
      case 2:
        return new PhraseQuery.Builder().add(t(f, "quick"), 0).add(t(f, "fox"), 2).build();
      case 3:
        return new MultiPhraseQuery.Builder()
            .add(new Term[] {t(f, "quick"), t(f, "lazy")})
            .add(new Term[] {t(f, "brown"), t(f, "dog")})
            .build();
      case 4:
        return bool(
            0,
            SHOULD,
            new SynonymQuery.Builder(f).addTerm(t(f, "fox")).addTerm(t(f, "quick")).build(),
            SHOULD,
            new PhraseQuery(f, "quick", "brown"));
      case 5:
        return new SpanNearQuery(
            new SpanQuery[] {
              new SpanMultiTermQueryWrapper<>(new PrefixQuery(t(f, "qu"))), st(f, "brown")
            },
            0,
            true);
      case 6:
        return new SpanOrQuery(
            st(f, "lazy"), new SpanMultiTermQueryWrapper<>(new WildcardQuery(t(f, "f?x"))));
      case 7:
        return bool(0, MUST, new TermQuery(t(f, "fox")), MUST, new TermQuery(t("id", "d0")));
      case 8:
        return bool(0, MUST, new TermQuery(t(f, "fox")), NOT, new TermQuery(t("id", "d4")));
      case 9:
        return bool(
            2,
            SHOULD,
            new TermQuery(t(f, "fox")),
            SHOULD,
            new TermQuery(t("id", "d1")),
            SHOULD,
            new TermQuery(t(f, "zzz")));
      case 10:
        return bool(
            2, SHOULD, new PhraseQuery(f, "quick", "brown"), SHOULD, new TermQuery(t("docs", "cat")));
      case 11:
        return new SpanNotQuery(st(f, "fox"), st(f, "quick"), 1, 1);
      case 12:
        return new SpanFirstQuery(st(f, "fox"), 5);
      case 13:
        return bool(0, SHOULD, new TermQuery(t(f, "fox")), SHOULD, new TermQuery(t("pos", "quick")));
      case 14:
        return new ConstantScoreQuery(
            new BoostQuery(
                new SpanNearQuery(new SpanQuery[] {st(f, "the"), st(f, "fox")}, 3, true), 2f));
      case 15:
        return new DisjunctionMaxQuery(
            List.of(new PhraseQuery(f, "lazy", "dog"), new TermQuery(t(f, "fox"))), 0.1f);
      case 16:
        return new SpanWithinQuery(
            new SpanNearQuery(new SpanQuery[] {st(f, "quick"), st(f, "fox")}, 3, true),
            st(f, "brown"));
      case 17:
        return new PhraseQuery(f, "fox");
      case 18:
        return bool(
            0,
            MUST,
            new SpanNearQuery(new SpanQuery[] {st(f, "quick"), st(f, "fox")}, 1, false),
            NOT,
            new TermQuery(t(f, "cat")));
      case 19:
        return bool(
            0, SHOULD, new PhraseQuery(f, "quick", "brown"), SHOULD, new PrefixQuery(t(f, "qu")));
      case 20:
        return new SpanNearQuery(
            new SpanQuery[] {
              new SpanMultiTermQueryWrapper<>(new RegexpQuery(t(f, "la.y"))), st(f, "dog")
            },
            0,
            true);
      case 21:
        return new SpanContainingQuery(
            new SpanNearQuery(new SpanQuery[] {st(f, "quick"), st(f, "fox")}, 2, true),
            st(f, "brown"));
      default:
        return bool(0, SHOULD, new PhraseQuery(f, "quick", "brown"), SHOULD, st("pos", "quick"));
    }
  }

  static final String[] BUILT_FIELDS = {"offs", "both", "tv", "pos", "docs"};

  /**
   * {@code uhighlight-built.tsv}: configuration, query number, field, {@code R}/{@code X}, then
   * the snippet of each document, for the hand-built queries under five configurations: the
   * default ({@code WEIGHT_MATCHES}), {@code PhraseHelper} with and without multi-term queries,
   * and both paths with a field matcher accepting every field.
   */
  static String built(IndexSearcher searcher, Analyzer analyzer) throws Exception {
    StringBuilder rows = new StringBuilder();
    int[] all = new int[searcher.getIndexReader().maxDoc()];
    for (int i = 0; i < all.length; i++) all[i] = i;
    for (int cfg = 0; cfg < 5; cfg++) {
      for (int qi = 0; qi < BUILT; qi++) {
        for (String field : BUILT_FIELDS) {
          UnifiedHighlighter.Builder b = UnifiedHighlighter.builder(searcher, analyzer);
          switch (cfg) {
            case 0:
              break;
            case 1:
              b.withWeightMatches(false);
              break;
            case 2:
              b.withWeightMatches(false).withHandleMultiTermQuery(false);
              break;
            case 3:
              b.withWeightMatches(false).withFieldMatcher(x -> true);
              break;
            default:
              b.withFieldMatcher(x -> true);
          }
          rows.append(cfg).append((char) 9).append(qi).append((char) 9).append(field);
          rows.append((char) 9);
          try {
            String[] snippets =
                b.build()
                    .highlightFields(new String[] {field}, built(qi, field), all, new int[] {2})
                    .get(field);
            StringBuilder joined = new StringBuilder();
            for (int i = 0; i < snippets.length; i++) {
              if (i > 0) joined.append((char) 31);
              joined.append(snippets[i] == null ? "\u0002" : snippets[i]);
            }
            rows.append("R").append((char) 9).append(esc(joined.toString()));
          } catch (Exception e) {
            rows.append("X").append((char) 9).append(esc(e.getClass().getName()));
          }
          rows.append((char) 10);
        }
      }
    }
    return rows.toString();
  }

  /** A snippet as an object: the number of passages and the default formatter's text. */
  record Snip(int passages, String text) {
    @Override
    public String toString() {
      return passages + "|" + text;
    }
  }

  /** Logs the content each {@code format} call is handed, in call order. */
  static final class LoggingFormatter extends PassageFormatter {
    final List<String> log = new ArrayList<>();
    final DefaultPassageFormatter inner = new DefaultPassageFormatter();

    @Override
    public Object format(Passage[] passages, String content) {
      log.add(content.substring(0, Math.min(12, content.length())));
      return new Snip(passages.length, inner.format(passages, content));
    }
  }

  /** {@code highlightFieldsAsObjects} is protected. */
  static final class ObjectsHighlighter extends UnifiedHighlighter {
    ObjectsHighlighter(Builder b) {
      super(b);
    }

    Map<String, Object[]> objects(String[] fields, Query q, int[] docs, int[] max)
        throws Exception {
      return highlightFieldsAsObjects(fields, q, docs, max);
    }
  }

  static final int[] CALL_THRESHOLDS = {-1, 0, 60};
  static final String[] CALL_FIELDS = {"tv", "offs", "pos", "both"};
  static final int[] CALL_MAX_PASSAGES = {1, 2, 1, 2};
  static final int[] CALL_DOCS = {5, 0, 3, 0, 7, 1};

  /** The two queries of {@code uhighlight-calls.tsv}: terms, then prefixes, on every field. */
  static Query callQuery(int i) {
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    for (String f : CALL_FIELDS) {
      b.add(i == 0 ? new TermQuery(t(f, "fox")) : new PrefixQuery(t(f, "qu")), SHOULD);
    }
    return b.build();
  }

  /**
   * {@code uhighlight-calls.tsv}: the stored-field batching ({@code cacheFieldValCharsThreshold},
   * -1 for the default) seen through the order of the formatter's calls, and {@code
   * highlightFieldsAsObjects}' objects: threshold, query, the call log, then per field (sorted)
   * the objects of each document of {@link #CALL_DOCS} ({@code \u0002} for none).
   */
  static String calls(IndexSearcher searcher, Analyzer analyzer) throws Exception {
    StringBuilder rows = new StringBuilder();
    for (int threshold : CALL_THRESHOLDS) {
      for (int qi = 0; qi < 2; qi++) {
        LoggingFormatter formatter = new LoggingFormatter();
        UnifiedHighlighter.Builder b =
            UnifiedHighlighter.builder(searcher, analyzer).withFormatter(formatter);
        if (threshold >= 0) b.withCacheFieldValCharsThreshold(threshold);
        Map<String, Object[]> got =
            new ObjectsHighlighter(b).objects(CALL_FIELDS, callQuery(qi), CALL_DOCS, CALL_MAX_PASSAGES);
        rows.append(threshold).append((char) 9).append(qi).append((char) 9);
        rows.append(esc(String.join("\u001f", formatter.log)));
        for (String field : new java.util.TreeSet<>(got.keySet())) {
          StringBuilder joined = new StringBuilder();
          Object[] objects = got.get(field);
          for (int i = 0; i < objects.length; i++) {
            if (i > 0) joined.append((char) 31);
            joined.append(objects[i] == null ? "\u0002" : objects[i].toString());
          }
          rows.append((char) 9).append(field).append('=').append(esc(joined.toString()));
        }
        rows.append((char) 10);
      }
    }
    return rows.toString();
  }

  static Supplier<BreakIterator> sentence() {
    return () -> BreakIterator.getSentenceInstance(Locale.ROOT);
  }

  /** The iterators of {@code breaks.tsv}, by number. */
  static BreakIterator iterator(int n) {
    switch (n) {
      case 0:
        return new WholeBreakIterator();
      case 1:
        return new CustomSeparatorBreakIterator(',');
      case 2:
        return new SplittingBreakIterator(sentence().get(), ';');
      case 3:
        return LengthGoalBreakIterator.createMinLength(sentence().get(), 20, 0.5f);
      case 4:
        return LengthGoalBreakIterator.createClosestToLength(sentence().get(), 20, 0.0f);
      case 5:
        return LengthGoalBreakIterator.createClosestToLength(
            BreakIterator.getWordInstance(Locale.ROOT), 10, 1.0f);
      default:
        return LengthGoalBreakIterator.createMinLength(
            new SplittingBreakIterator(sentence().get(), ';'), 30, 0.25f);
    }
  }

  /** {@code iterator \t text \t op:result ...} for random texts and operations. */
  static String breaks() {
    String[] alphabet = {
      "a", "b", "fox", "quick", " ", " ", ".", ". ", ",", ";", "!", "?", "x", "\n", "1", "é"
    };
    Random rnd = new Random(31415L);
    StringBuilder sb = new StringBuilder();
    for (int row = 0; row < 3000; row++) {
      int it = row % 7;
      StringBuilder t = new StringBuilder();
      int len = rnd.nextInt(24);
      for (int i = 0; i < len; i++) t.append(alphabet[rnd.nextInt(alphabet.length)]);
      String text = t.toString();
      BreakIterator bi = iterator(it);
      bi.setText(text);
      StringBuilder ops = new StringBuilder();
      int n = 1 + rnd.nextInt(10);
      for (int i = 0; i < n; i++) {
        int op = rnd.nextInt(it >= 3 && it <= 6 ? 6 : 7);
        int off = rnd.nextInt(text.length() + 1);
        String r;
        switch (op) {
          case 0:
            r = "F" + bi.first();
            break;
          case 1:
            r = "L" + bi.last();
            break;
          case 2:
            r = "N" + bi.next();
            break;
          case 3:
            r = "f" + off + "=" + bi.following(off);
            break;
          case 4:
            r = "p" + off + "=" + bi.preceding(off);
            break;
          case 5:
            r = "C" + bi.current();
            break;
          default:
            r = "P" + bi.previous();
        }
        if (i > 0) ops.append(' ');
        ops.append(r);
      }
      sb.append(it).append((char) 9).append(esc(text)).append((char) 9).append(ops).append((char) 10);
    }
    return sb.toString();
  }
}
