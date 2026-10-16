import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
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
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.matchhighlight.FieldValueHighlighters;
import org.apache.lucene.search.matchhighlight.MatchHighlighter;
import org.apache.lucene.search.matchhighlight.PassageFormatter;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.2 ({@code matchhighlight}): an index of {@code fixtures/corpus/matchhighlight-docs.tsv}
 * (stored multi-valued fields indexed with offsets, positions only, and docs only), and what
 * {@code MatchHighlighter} makes of every query of {@code matchhighlight-queries.txt} under
 * three highlighter configurations. {@code matchhighlight.tsv}: configuration, query index,
 * then per hit document in the order returned: {@code id=field:values|...}. {@code words.tsv}:
 * the boundaries of {@code BreakIterator.getWordInstance(Locale.ROOT)} over
 * {@code matchhighlight-words.txt}. {@code crates/lucene-search/tests/matchhighlight_fixtures.rs}
 * compares.
 */
public class GenMatchHighlight {

  static String esc(String s) {
    return AnalysisRows.esc(s);
  }

  static FieldType type(IndexOptions o) {
    FieldType t = new FieldType();
    t.setIndexOptions(o);
    t.setTokenized(true);
    t.setStored(true);
    t.freeze();
    return t;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("matchhighlight");
    Files.createDirectories(out);
    Analyzer analyzer = new StandardAnalyzer();

    Path indexDir = out.resolve("index");
    if (Files.exists(indexDir)) {
      try (var walk = Files.walk(indexDir)) {
        walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    FieldType offs = type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    FieldType pos = type(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    FieldType docs = type(IndexOptions.DOCS);
    try (FSDirectory fs = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(fs, new IndexWriterConfig(analyzer).setUseCompoundFile(false))) {
      for (String line : AnalysisRows.corpus("matchhighlight-docs.tsv")) {
        String[] c = line.split("\t", -1);
        Document d = new Document();
        d.add(new Field("id", c[0], org.apache.lucene.document.StringField.TYPE_STORED));
        for (String v : c[1].split("\u0001")) d.add(new Field("offs", v, offs));
        for (String v : c[2].split("\u0001")) d.add(new Field("pos", v, pos));
        for (String v : c[3].split("\u0001")) d.add(new Field("docs", v, docs));
        for (String v : c[4].split("\u0001")) d.add(new Field("title", v, pos));
        w.addDocument(d);
      }
      w.forceMerge(1);
    }

    StringBuilder rows = new StringBuilder();
    try (FSDirectory fs = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(fs)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      QueryParser parser = new QueryParser("offs", analyzer);
      List<String> queries = AnalysisRows.corpus("matchhighlight-queries.txt");
      for (int qi = 0; qi < queries.size(); qi++) {
        Query q = parser.parse(queries.get(qi));
        TopDocs top = searcher.search(q, 20);
        for (int cfg = 0; cfg < 3; cfg++) {
          MatchHighlighter mh = new MatchHighlighter(searcher, analyzer);
          switch (cfg) {
            case 0:
              mh.appendFieldHighlighter(
                  FieldValueHighlighters.highlighted(
                      32, 2, new PassageFormatter("...", "<em>", "</em>"), f -> !f.equals("id") && !f.equals("title")));
              mh.appendFieldHighlighter(FieldValueHighlighters.verbatimValue("id"));
              mh.appendFieldHighlighter(FieldValueHighlighters.skipRemaining());
              break;
            case 1:
              mh.appendFieldHighlighter(
                  FieldValueHighlighters.highlighted(
                      80, 3, new PassageFormatter(" … ", r -> "[" + r.from + ":", r -> "]"), f -> true));
              break;
            default:
              mh.appendFieldHighlighter(FieldValueHighlighters.maxLeadingCharacters(40, "…", Set.of("title")));
              mh.appendFieldHighlighter(
                  FieldValueHighlighters.highlighted(
                      16, 4, new PassageFormatter("~", "(", ")"), f -> f.equals("offs") || f.equals("pos")));
              mh.appendFieldHighlighter(FieldValueHighlighters.verbatimValue("id"));
              mh.appendFieldHighlighter(FieldValueHighlighters.skipRemaining());
          }
          rows.append(cfg).append((char) 9).append(qi).append((char) 9);
          try {
            List<String> hits =
                mh.highlight(top, q)
                    .map(
                        dh ->
                            dh.docId
                                + "="
                                + dh.fields.entrySet().stream()
                                    .map(e -> e.getKey() + ":" + String.join((char) 31 + "", e.getValue()))
                                    .collect(Collectors.joining("|")))
                    .collect(Collectors.toList());
            rows.append("R").append((char) 9).append(esc(String.join(String.valueOf((char) 10), hits)));
          } catch (Exception e) {
            rows.append("X").append((char) 9).append(esc(e.getClass().getName() + ": " + e.getMessage()));
          }
          rows.append((char) 10);
        }
      }
    }
    Files.writeString(out.resolve("matchhighlight.tsv"), rows.toString(), StandardCharsets.UTF_8);

    StringBuilder words = new StringBuilder();
    BreakIterator bi = BreakIterator.getWordInstance(Locale.ROOT);
    for (String raw : AnalysisRows.corpus("matchhighlight-words.txt")) {
      String text = raw.replace("\t", "\t").replace("\n", "\n");
      bi.setText(text);
      List<String> bounds = new ArrayList<>();
      for (int b = bi.first(); b != BreakIterator.DONE; b = bi.next()) bounds.add(Integer.toString(b));
      words.append(esc(text)).append((char) 9).append(String.join(",", bounds)).append((char) 10);
    }
    Files.writeString(out.resolve("words.tsv"), words.toString(), StandardCharsets.UTF_8);

    String[] alphabet = {
      "a", "Z", "q", "\u00e9", "\u65e5", "\u3005", "\u3042", "\u30a2", "1", "7", "\u0663", ".", ",", "'", "\"", "-", "_",
      "%", "$", "#", "&", " ", " ", "\t", "\n", "\r", "\f", "\u0301", "\u00ad", "\u200b", "\u3099", "\u30fb", "\u30fc",
      ":", "!", "\u20ac", "\u0964", "\u2028", "\u00a0", "\u066b", "\u00a2", "@", "\u0000", "\ud83d\ude00", "\ud835\udc00",
      "\ud801\udc00", "\ud835\udfce", "\u0e01", "\uac00", "\u1100", "\ud800", "\udc00", "\u0903", "(", "/"
    };
    Files.writeString(out.resolve("wordfuzz.tsv"), fuzz(() -> BreakIterator.getWordInstance(Locale.ROOT), alphabet, 20251016L), StandardCharsets.UTF_8);
    String[] sentenceAlphabet = {
      "a", "B", "z", "\u00e9", "\u65e5", "1", "9", ".", "!", "?", ",", "'", "\"", " ", " ", "\n", "\t", "\r",
      "\u2028", "\u2029", "\u3002", "\uff0e", "\uff01", "\uff1f", "\u0964", "(", ")", "[", "]", "\u201c", "\u201d",
      "\u00ab", "\u00bb", "\u0301", "\u200b", "\u00a0", "-", ":", ";", "\ud83d\ude00", "\ud801\udc00", "\u0663", "\u3042", "\u01c5", "\u02b0"
    };
    Files.writeString(out.resolve("sentencefuzz.tsv"), fuzz(() -> BreakIterator.getSentenceInstance(Locale.ROOT), sentenceAlphabet, 77L), StandardCharsets.UTF_8);
  }

  static String fuzz(java.util.function.Supplier<BreakIterator> mk, String[] alphabet, long seed) {
    java.util.Random rnd = new java.util.Random(seed);
    StringBuilder fuzz = new StringBuilder();
    BreakIterator shared = mk.get();
    for (int n = 0; n < 4000; n++) {
      StringBuilder sb = new StringBuilder();
      int len = rnd.nextInt(n < 200 ? 4 : 22);
      for (int i = 0; i < len; i++) sb.append(alphabet[rnd.nextInt(alphabet.length)]);
      String text = sb.toString();
      BreakIterator it = mk.get();
      it.setText(text);
      List<String> next = new ArrayList<>();
      for (int b = it.first(); b != BreakIterator.DONE; b = it.next()) next.add(Integer.toString(b));
      List<String> fresh = new ArrayList<>();
      for (int off = 0; off <= text.length(); off++) {
        BreakIterator f = mk.get();
        f.setText(text);
        boolean isb = f.isBoundary(off);
        BreakIterator g = mk.get();
        g.setText(text);
        int fol = g.following(off);
        BreakIterator h = mk.get();
        h.setText(text);
        int pre = h.preceding(off);
        fresh.add(off + ":" + isb + "," + fol + "," + pre);
      }
      List<Integer> order = new ArrayList<>();
      for (int off = 0; off <= text.length(); off++) order.add(off);
      java.util.Collections.shuffle(order, rnd);
      shared.setText(text);
      List<String> seq = new ArrayList<>();
      for (int off : order) {
        int kind = rnd.nextInt(3);
        if (kind == 0) seq.add(off + "b" + shared.isBoundary(off));
        else if (kind == 1) seq.add(off + "f" + shared.following(off));
        else seq.add(off + "p" + shared.preceding(off));
      }
      fuzz.append(esc(text)).append((char) 9).append(String.join(",", next)).append((char) 9)
          .append(String.join(" ", fresh)).append((char) 9).append(String.join(" ", seq)).append((char) 10);
    }
    return fuzz.toString();
  }
}
