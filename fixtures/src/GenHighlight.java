import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;
import org.apache.lucene.queries.spans.SpanFirstQuery;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanNotQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.highlight.DefaultEncoder;
import org.apache.lucene.search.highlight.Encoder;
import org.apache.lucene.search.highlight.Formatter;
import org.apache.lucene.search.highlight.Fragmenter;
import org.apache.lucene.search.highlight.GradientFormatter;
import org.apache.lucene.search.highlight.Highlighter;
import org.apache.lucene.search.highlight.NullFragmenter;
import org.apache.lucene.search.highlight.QueryScorer;
import org.apache.lucene.search.highlight.QueryTermScorer;
import org.apache.lucene.search.highlight.SimpleFragmenter;
import org.apache.lucene.search.highlight.SimpleHTMLEncoder;
import org.apache.lucene.search.highlight.SimpleHTMLFormatter;
import org.apache.lucene.search.highlight.SimpleSpanFragmenter;
import org.apache.lucene.search.highlight.SpanGradientFormatter;
import org.apache.lucene.search.highlight.TextFragment;
import org.apache.lucene.search.highlight.WeightedSpanTerm;

/**
 * M13 T13.2 (classic {@code Highlighter}): every query of {@code
 * fixtures/corpus/highlight-queries.txt} (classic syntax over {@code body}; a {@code span:} line is
 * an s-expression of span queries) against every text of {@code highlight-texts.txt}, under a set
 * of configurations (scorer, fragmenter, formatter, encoder, fragment count, character limit).
 * {@code highlight.tsv}: configuration, query index, text index, then the highlighted fragments
 * (escaped, joined by {@code <|>}) or {@code X} and the exception. {@code spanterms.tsv}: what
 * {@code WeightedSpanTermExtractor} extracts, per query and text.
 * {@code crates/lucene-search/tests/highlight_fixtures.rs} compares.
 */
public class GenHighlight {

  static String esc(String s) {
    return AnalysisRows.esc(s);
  }

  /** The {@code span:} s-expressions: {@code (term t)}, {@code (near slop ordered a b ..)}, ... */
  static SpanQuery span(List<String> toks, int[] pos) {
    String open = toks.get(pos[0]++);
    if (!open.equals("(")) throw new IllegalStateException(open);
    String kind = toks.get(pos[0]++);
    SpanQuery result;
    switch (kind) {
      case "term":
        result = new SpanTermQuery(new Term("body", toks.get(pos[0]++)));
        break;
      case "near":
        {
          int slop = Integer.parseInt(toks.get(pos[0]++));
          boolean ordered = Boolean.parseBoolean(toks.get(pos[0]++));
          List<SpanQuery> cs = new ArrayList<>();
          while (!toks.get(pos[0]).equals(")")) cs.add(span(toks, pos));
          result = new SpanNearQuery(cs.toArray(new SpanQuery[0]), slop, ordered);
          break;
        }
      case "or":
        {
          List<SpanQuery> cs = new ArrayList<>();
          while (!toks.get(pos[0]).equals(")")) cs.add(span(toks, pos));
          result = new SpanOrQuery(cs.toArray(new SpanQuery[0]));
          break;
        }
      case "first":
        {
          int end = Integer.parseInt(toks.get(pos[0]++));
          result = new SpanFirstQuery(span(toks, pos), end);
          break;
        }
      case "not":
        {
          SpanQuery inc = span(toks, pos);
          SpanQuery exc = span(toks, pos);
          result = new SpanNotQuery(inc, exc);
          break;
        }
      default:
        throw new IllegalStateException(kind);
    }
    if (!toks.get(pos[0]++).equals(")")) throw new IllegalStateException("close");
    return result;
  }

  static Query parse(QueryParser parser, String line) throws Exception {
    if (line.startsWith("span: ")) {
      List<String> toks = new ArrayList<>();
      for (String t : line.substring(6).replace("(", " ( ").replace(")", " ) ").trim().split("\\s+")) {
        toks.add(t);
      }
      return span(toks, new int[] {0});
    }
    return parser.parse(line);
  }

  interface Config {
    String run(Query q, Analyzer analyzer, String text) throws Exception;
  }

  static String join(String[] frags) {
    return String.join("<|>", frags);
  }

  static String texts(TextFragment[] frags) {
    List<String> out = new ArrayList<>();
    for (TextFragment f : frags) {
      out.add(Float.toString(f.getScore()) + ":" + f.getFragNum() + ":" + f);
    }
    return String.join("<|>", out);
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("highlight");
    Files.createDirectories(out);
    List<String> queries = AnalysisRows.corpus("highlight-queries.txt");
    List<String> texts = AnalysisRows.corpus("highlight-texts.txt");
    Analyzer analyzer = new StandardAnalyzer();
    QueryParser parser = new QueryParser("body", analyzer);

    // the idf-weighted scorers read document frequencies from an index of the same texts
    org.apache.lucene.store.ByteBuffersDirectory dir = new org.apache.lucene.store.ByteBuffersDirectory();
    try (org.apache.lucene.index.IndexWriter w =
        new org.apache.lucene.index.IndexWriter(dir, new org.apache.lucene.index.IndexWriterConfig(analyzer))) {
      for (String t : texts) {
        org.apache.lucene.document.Document d = new org.apache.lucene.document.Document();
        d.add(new org.apache.lucene.document.TextField("body", t, org.apache.lucene.document.Field.Store.NO));
        w.addDocument(d);
      }
    }
    final org.apache.lucene.index.IndexReader reader = org.apache.lucene.index.DirectoryReader.open(dir);

    Map<String, Config> configs = new TreeMap<>();
    configs.put(
        "L",
        (q, a, t) -> {
          Highlighter h = new Highlighter(new QueryScorer(q, reader, "body"));
          h.setTextFragmenter(new SimpleFragmenter(40));
          return texts(h.getBestTextFragments(a.tokenStream("body", t), t, true, 3));
        });
    configs.put(
        "M",
        (q, a, t) -> {
          Highlighter h = new Highlighter(new QueryTermScorer(q, reader, "body"));
          h.setTextFragmenter(new SimpleFragmenter(40));
          return texts(h.getBestTextFragments(a.tokenStream("body", t), t, true, 3));
        });
    configs.put(
        "A",
        (q, a, t) -> {
          Highlighter h = new Highlighter(new QueryScorer(q, "body"));
          h.setTextFragmenter(new SimpleFragmenter(40));
          return join(h.getBestFragments(a.tokenStream("body", t), t, 3));
        });
    configs.put(
        "B",
        (q, a, t) -> {
          QueryScorer s = new QueryScorer(q, "body");
          Highlighter h = new Highlighter(s);
          h.setTextFragmenter(new SimpleSpanFragmenter(s, 40));
          return join(h.getBestFragments(a.tokenStream("body", t), t, 3));
        });
    configs.put(
        "C",
        (q, a, t) -> {
          Highlighter h = new Highlighter(new QueryTermScorer(q, "body"));
          h.setTextFragmenter(new SimpleFragmenter(30));
          String f = h.getBestFragment(a.tokenStream("body", t), t);
          return f == null ? "" : f;
        });
    configs.put(
        "D",
        (q, a, t) -> {
          Highlighter h =
              new Highlighter(new SimpleHTMLFormatter("[", "]"), new SimpleHTMLEncoder(), new QueryScorer(q));
          h.setTextFragmenter(new SimpleFragmenter(25));
          return join(h.getBestFragments(a.tokenStream("body", t), t, 5));
        });
    configs.put(
        "E",
        (q, a, t) -> {
          Highlighter h = new Highlighter(new QueryScorer(q, "body"));
          h.setTextFragmenter(new NullFragmenter());
          return join(h.getBestFragments(a.tokenStream("body", t), t, 1));
        });
    configs.put(
        "F",
        (q, a, t) -> {
          Highlighter h = new Highlighter(new QueryScorer(q, "body"));
          h.setTextFragmenter(new SimpleFragmenter(20));
          h.setMaxDocCharsToAnalyze(30);
          return join(h.getBestFragments(a.tokenStream("body", t), t, 4));
        });
    configs.put(
        "G",
        (q, a, t) -> {
          Highlighter h = new Highlighter(new QueryScorer(q, "body"));
          h.setTextFragmenter(new SimpleFragmenter(35));
          return texts(h.getBestTextFragments(a.tokenStream("body", t), t, false, 4));
        });
    configs.put(
        "H",
        (q, a, t) -> {
          QueryScorer s = new QueryScorer(q, "body");
          Highlighter h =
              new Highlighter(new GradientFormatter(2f, "#000000", "#FF0000", "#FFFFFF", "#00FF00"), s);
          h.setTextFragmenter(new SimpleFragmenter(60));
          return join(h.getBestFragments(a.tokenStream("body", t), t, 2));
        });
    configs.put(
        "I",
        (q, a, t) -> {
          QueryScorer s = new QueryScorer(q, "body");
          Highlighter h =
              new Highlighter(new SpanGradientFormatter(2f, null, null, "#112233", "#AABBCC"), s);
          h.setTextFragmenter(new SimpleFragmenter(60));
          return join(h.getBestFragments(a.tokenStream("body", t), t, 2));
        });
    configs.put(
        "J",
        (q, a, t) -> {
          QueryScorer s = new QueryScorer(q, "body");
          s.setExpandMultiTermQuery(false);
          Highlighter h = new Highlighter(s);
          h.setTextFragmenter(new SimpleFragmenter(40));
          return join(h.getBestFragments(a.tokenStream("body", t), t, 3));
        });
    configs.put(
        "K",
        (q, a, t) -> {
          Highlighter h = new Highlighter(new QueryScorer(q, "title", "body"));
          h.setTextFragmenter(new SimpleFragmenter(40));
          return h.getBestFragments(a.tokenStream("body", t), t, 2, "...");
        });

    StringBuilder rows = new StringBuilder();
    StringBuilder spans = new StringBuilder();
    for (int qi = 0; qi < queries.size(); qi++) {
      Query q = parse(parser, queries.get(qi));
      for (int ti = 0; ti < texts.size(); ti++) {
        String t = texts.get(ti);
        for (Map.Entry<String, Config> c : configs.entrySet()) {
          rows.append(c.getKey()).append((char) 9).append(qi).append((char) 9).append(ti).append((char) 9);
          try {
            rows.append("R").append((char) 9).append(esc(c.getValue().run(q, analyzer, t)));
          } catch (Exception e) {
            rows.append("X").append((char) 9).append(esc(e.getClass().getName() + ": " + e.getMessage()));
          }
          rows.append((char) 10);
        }
        // the weighted span terms the scorer extracts
        spans.append(qi).append((char) 9).append(ti).append((char) 9);
        try (TokenStream ts = analyzer.tokenStream("body", t)) {
          Map<String, WeightedSpanTerm> terms =
              org.apache.lucene.search.highlight.HighlightFixtureAccess.extract(q, ts, "body", 50 * 1024);
          List<String> cells = new ArrayList<>();
          for (Map.Entry<String, WeightedSpanTerm> e : new TreeMap<>(terms).entrySet()) {
            WeightedSpanTerm w = e.getValue();
            List<String> ps = new ArrayList<>();
            for (var p : w.getPositionSpans()) {
              ps.add(spanPos(p));
            }
            Collections.sort(ps);
            cells.add(e.getKey() + "=" + w.getWeight() + "=" + w.isPositionSensitive() + "=" + String.join(",", ps));
          }
          spans.append("R").append((char) 9).append(esc(String.join(" ", cells)));
        } catch (Exception e) {
          spans.append("X").append((char) 9).append(esc(e.getClass().getName() + ": " + e.getMessage()));
        }
        spans.append((char) 10);
      }
    }
    // term vectors: an index (written to disk, for the Rust reader) of the texts in three fields
    // -- positions and offsets, offsets only, positions only -- and what TokenSources makes of each
    Path tvIndex = out.resolve("tvindex");
    if (Files.exists(tvIndex)) {
      try (var walk = Files.walk(tvIndex)) {
        walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    org.apache.lucene.document.FieldType full = new org.apache.lucene.document.FieldType();
    full.setIndexOptions(org.apache.lucene.index.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    full.setTokenized(true);
    full.setStoreTermVectors(true);
    full.setStoreTermVectorPositions(true);
    full.setStoreTermVectorOffsets(true);
    full.freeze();
    org.apache.lucene.document.FieldType offs = new org.apache.lucene.document.FieldType();
    offs.setIndexOptions(org.apache.lucene.index.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    offs.setTokenized(true);
    offs.setStoreTermVectors(true);
    offs.setStoreTermVectorOffsets(true);
    offs.freeze();
    org.apache.lucene.document.FieldType poss = new org.apache.lucene.document.FieldType();
    poss.setIndexOptions(org.apache.lucene.index.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    poss.setTokenized(true);
    poss.setStoreTermVectors(true);
    poss.setStoreTermVectorPositions(true);
    poss.freeze();
    try (org.apache.lucene.store.FSDirectory fs = org.apache.lucene.store.FSDirectory.open(tvIndex);
        org.apache.lucene.index.IndexWriter w =
            new org.apache.lucene.index.IndexWriter(fs, new org.apache.lucene.index.IndexWriterConfig(analyzer).setUseCompoundFile(false))) {
      for (String t : texts) {
        org.apache.lucene.document.Document d = new org.apache.lucene.document.Document();
        d.add(new org.apache.lucene.document.Field("tv_full", t, full));
        d.add(new org.apache.lucene.document.Field("tv_off", t, offs));
        d.add(new org.apache.lucene.document.Field("tv_pos", t, poss));
        w.addDocument(d);
      }
      w.forceMerge(1);
    }
    StringBuilder tvRows = new StringBuilder();
    StringBuilder tvHl = new StringBuilder();
    try (org.apache.lucene.store.FSDirectory fs = org.apache.lucene.store.FSDirectory.open(tvIndex);
        org.apache.lucene.index.DirectoryReader r = org.apache.lucene.index.DirectoryReader.open(fs)) {
      for (String field : new String[] {"tv_full", "tv_off", "tv_pos"}) {
        for (int doc = 0; doc < texts.size(); doc++) {
          org.apache.lucene.index.Fields fields = r.termVectors().get(doc);
          for (int max : new int[] {-1, 20, 0}) {
            tvRows.append(field).append((char) 9).append(doc).append((char) 9).append(max).append((char) 9);
            try (TokenStream ts = org.apache.lucene.search.highlight.TokenSources.getTokenStream(field, fields, texts.get(doc), analyzer, max)) {
              var term = ts.addAttribute(org.apache.lucene.analysis.tokenattributes.CharTermAttribute.class);
              var off = ts.addAttribute(org.apache.lucene.analysis.tokenattributes.OffsetAttribute.class);
              var inc = ts.addAttribute(org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute.class);
              List<String> toks = new ArrayList<>();
              ts.reset();
              while (ts.incrementToken()) {
                toks.add(term + "@" + off.startOffset() + "-" + off.endOffset() + "+" + inc.getPositionIncrement());
              }
              ts.end();
              tvRows.append("R").append((char) 9).append(esc(String.join(" ", toks)));
            } catch (Exception e) {
              tvRows.append("X").append((char) 9).append(esc(e.getClass().getName() + ": " + e.getMessage()));
            }
            tvRows.append((char) 10);
          }
          // the tokens of the term vector alone: null where it has no offsets
          tvRows.append(field).append((char) 9).append(doc).append((char) 9).append("only").append((char) 9);
          TokenStream only = org.apache.lucene.search.highlight.TokenSources.getTermVectorTokenStreamOrNull(field, fields, -1);
          tvRows.append(only == null ? "null" : "stream").append((char) 10);
        }
      }
      // highlighting through the term vector's stream
      for (int qi = 0; qi < queries.size(); qi++) {
        Query q = parse(parser, queries.get(qi));
        for (int doc = 0; doc < texts.size(); doc++) {
          for (String field : new String[] {"tv_full", "tv_off"}) {
            org.apache.lucene.index.Fields fields = r.termVectors().get(doc);
            tvHl.append(field).append((char) 9).append(qi).append((char) 9).append(doc).append((char) 9);
            try {
              TokenStream ts = org.apache.lucene.search.highlight.TokenSources.getTokenStream(field, fields, texts.get(doc), analyzer, -1);
              Highlighter h = new Highlighter(new QueryScorer(q, "body"));
              h.setTextFragmenter(new SimpleFragmenter(40));
              tvHl.append("R").append((char) 9).append(esc(join(h.getBestFragments(ts, texts.get(doc), 3))));
            } catch (Exception e) {
              tvHl.append("X").append((char) 9).append(esc(e.getClass().getName() + ": " + e.getMessage()));
            }
            tvHl.append((char) 10);
          }
        }
      }
    }
    Files.writeString(out.resolve("termvectors.tsv"), tvRows.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("tv_highlight.tsv"), tvHl.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("highlight.tsv"), rows.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("spanterms.tsv"), spans.toString(), StandardCharsets.UTF_8);
  }

  /** {@code PositionSpan}'s fields are package-private: read them reflectively. */
  static String spanPos(Object p) throws Exception {
    var s = p.getClass().getDeclaredField("start");
    var e = p.getClass().getDeclaredField("end");
    s.setAccessible(true);
    e.setAccessible(true);
    return s.get(p) + "-" + e.get(p);
  }
}
