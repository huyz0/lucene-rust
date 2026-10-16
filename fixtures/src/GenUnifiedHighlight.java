import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
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
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.uhighlight.CustomSeparatorBreakIterator;
import org.apache.lucene.search.uhighlight.DefaultPassageFormatter;
import org.apache.lucene.search.uhighlight.LengthGoalBreakIterator;
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
 * {@code breaks.tsv}: random texts through Lucene's break iterators. {@code
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
