import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.BreakIterator;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.DelegatingAnalyzerWrapper;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.SynonymQuery;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.highlight.DefaultEncoder;
import org.apache.lucene.search.highlight.Encoder;
import org.apache.lucene.search.highlight.SimpleHTMLEncoder;
import org.apache.lucene.search.vectorhighlight.BaseFragmentsBuilder;
import org.apache.lucene.search.vectorhighlight.BreakIteratorBoundaryScanner;
import org.apache.lucene.search.vectorhighlight.FastVectorHighlighter;
import org.apache.lucene.search.vectorhighlight.FieldQuery;
import org.apache.lucene.search.vectorhighlight.FragListBuilder;
import org.apache.lucene.search.vectorhighlight.FragmentsBuilder;
import org.apache.lucene.search.vectorhighlight.ScoreOrderFragmentsBuilder;
import org.apache.lucene.search.vectorhighlight.SimpleBoundaryScanner;
import org.apache.lucene.search.vectorhighlight.SimpleFragListBuilder;
import org.apache.lucene.search.vectorhighlight.SimpleFragmentsBuilder;
import org.apache.lucene.search.vectorhighlight.SingleFragListBuilder;
import org.apache.lucene.search.vectorhighlight.WeightedFragListBuilder;
import org.apache.lucene.store.FSDirectory;

/**
 * M13 T13.2 ({@code vectorhighlight}): the docs of {@code matchhighlight-docs.tsv} (and a few
 * more) indexed with term vectors (positions and offsets) in {@code tv}, the same text analysed
 * without stop words in {@code tvb}, term vectors without offsets in {@code tvpos}, without
 * positions in {@code tvterms}, with a token over two ({@link QbFilter}) in {@code tvsyn}, and no
 * term vectors in {@code plain}; what {@code FastVectorHighlighter} makes of every query of {@code
 * vhighlight-queries.txt} ({@code F:} stands for the field) and of ten queries built by hand,
 * under eight configurations, over every document. {@code vhighlight.tsv}: configuration, query,
 * field, document, {@code R}/{@code N}/
 * {@code X}, then the fragments. {@code crates/lucene-search/tests/vhighlight_fixtures.rs}
 * compares.
 */
public class GenVectorHighlight {

  static String esc(String s) {
    return AnalysisRows.esc(s);
  }

  static FieldType type(boolean tv, boolean offsets) {
    FieldType t = new FieldType();
    t.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    t.setTokenized(true);
    t.setStored(true);
    if (tv) {
      t.setStoreTermVectors(true);
      t.setStoreTermVectorPositions(true);
      t.setStoreTermVectorOffsets(offsets);
    }
    t.freeze();
    return t;
  }

  static final String[] FIELDS = {"tv", "tvb", "tvpos", "plain", "tvsyn", "tvterms"};

  /**
   * After each {@code quick}, a token {@code qb} at the same position whose offsets run eleven
   * characters on (over {@code quick brown}): same-position occurrences and phrases that overlap.
   */
  static final class QbFilter extends TokenFilter {
    final CharTermAttribute term = addAttribute(CharTermAttribute.class);
    final OffsetAttribute offset = addAttribute(OffsetAttribute.class);
    final PositionIncrementAttribute inc = addAttribute(PositionIncrementAttribute.class);
    State pending;

    QbFilter(TokenStream in) {
      super(in);
    }

    @Override
    public boolean incrementToken() throws IOException {
      if (pending != null) {
        restoreState(pending);
        pending = null;
        int start = offset.startOffset();
        term.setEmpty().append("qb");
        offset.setOffset(start, start + 11);
        inc.setPositionIncrement(0);
        return true;
      }
      if (!input.incrementToken()) return false;
      if (term.toString().equals("quick")) pending = captureState();
      return true;
    }

    @Override
    public void reset() throws IOException {
      super.reset();
      pending = null;
    }
  }

  static Term t(String text) {
    return new Term("tv", text);
  }

  /** The queries the classic parser cannot write, by number past the corpus's. */
  static final int BUILT = 10;

  static Query built(int i) {
    switch (i) {
      case 0:
        return new DisjunctionMaxQuery(
            List.of(new TermQuery(t("fox")), new PhraseQuery("tv", "lazy", "dog")), 0.1f);
      case 1:
        return new SynonymQuery.Builder("tv").addTerm(t("fox")).addTerm(t("dog"), 0.5f).build();
      case 2:
        return new ConstantScoreQuery(new TermQuery(t("quick")));
      case 3:
        return new MultiPhraseQuery.Builder().add(new Term[] {t("quick"), t("lazy")}).build();
      case 4:
        return new MultiPhraseQuery.Builder().add(t("quick")).add(t("brown")).build();
      case 5:
        return new TermInSetQuery("tv", List.of(new BytesRef("fox"), new BytesRef("dog"), new BytesRef("zzz")));
      case 6:
        return new BoostQuery(
            new BooleanQuery.Builder()
                .add(new TermQuery(t("fox")), BooleanClause.Occur.SHOULD)
                .add(new TermQuery(t("cat")), BooleanClause.Occur.MUST_NOT)
                .build(),
            2f);
      case 7:
        return new PhraseQuery("tv", "fox");
      case 8:
        // on tvsyn, qb comes before quick at its position: neither is the start of a phrase
        return new BooleanQuery.Builder()
            .add(new PhraseQuery("tvsyn", "the", "quick"), BooleanClause.Occur.SHOULD)
            .add(new PhraseQuery("tvsyn", "fox", "qb"), BooleanClause.Occur.SHOULD)
            .build();
      default:
        return new BooleanQuery.Builder()
            .add(new BoostQuery(new PhraseQuery("tv", "quick", "brown"), 2f), BooleanClause.Occur.SHOULD)
            .add(new PhraseQuery(1, "tv", "brown", "fox"), BooleanClause.Occur.SHOULD)
            .add(new PhraseQuery("tv", "brown", "fox"), BooleanClause.Occur.SHOULD)
            .build();
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("vhighlight");
    Files.createDirectories(out);
    Analyzer standard = new StandardAnalyzer();
    Analyzer noStop = new StandardAnalyzer(CharArraySet.EMPTY_SET);
    Analyzer syn =
        new Analyzer() {
          @Override
          protected TokenStreamComponents createComponents(String fieldName) {
            org.apache.lucene.analysis.standard.StandardTokenizer src =
                new org.apache.lucene.analysis.standard.StandardTokenizer();
            TokenStream ts = new org.apache.lucene.analysis.LowerCaseFilter(src);
            return new TokenStreamComponents(src, new QbFilter(ts));
          }
        };
    Analyzer analyzer =
        new DelegatingAnalyzerWrapper(Analyzer.PER_FIELD_REUSE_STRATEGY) {
          @Override
          protected Analyzer getWrappedAnalyzer(String fieldName) {
            return fieldName.equals("tvb") ? noStop : fieldName.equals("tvsyn") ? syn : standard;
          }
        };

    Path indexDir = out.resolve("index");
    if (Files.exists(indexDir)) {
      try (var walk = Files.walk(indexDir)) {
        walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    FieldType full = type(true, true);
    FieldType noOffsets = type(true, false);
    FieldType plain = type(false, false);
    FieldType termsOnly = new FieldType(plain);
    termsOnly.setStoreTermVectors(true);
    termsOnly.freeze();
    try (FSDirectory fs = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(fs, new IndexWriterConfig(analyzer).setUseCompoundFile(false))) {
      List<String[]> docs = new java.util.ArrayList<>();
      for (String line : AnalysisRows.corpus("matchhighlight-docs.tsv")) {
        docs.add(line.split("\t", -1)[1].split("\u0001"));
      }
      // empty values between the others
      docs.add(new String[] {"", "fox one", "", "quick brown fox and the lazy dog", "dog"});
      // no field at all, so no term vectors
      docs.add(new String[] {});
      // qb's offsets run from one value into the next, and over a whole one
      docs.add(new String[] {"a quick", "brown fox jumps"});
      docs.add(new String[] {"quick", "x", "brown fox"});
      // more terms than a multi-term query expands to
      StringBuilder many = new StringBuilder();
      for (int i = 0; i < 1100; i++) many.append(String.format(Locale.ROOT, "t%04d ", i));
      docs.add(new String[] {many.toString().trim()});
      for (String[] values : docs) {
        Document d = new Document();
        for (String v : values) {
          d.add(new Field("tv", v, full));
          d.add(new Field("tvb", v, full));
          d.add(new Field("tvpos", v, noOffsets));
          d.add(new Field("plain", v, plain));
          d.add(new Field("tvsyn", v, full));
          d.add(new Field("tvterms", v, termsOnly));
        }
        w.addDocument(d);
      }
      w.forceMerge(1);
    }

    StringBuilder rows = new StringBuilder();
    try (FSDirectory fs = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(fs)) {
      QueryParser parser = new QueryParser("tv", standard);
      List<String> queries = AnalysisRows.corpus("vhighlight-queries.txt");
      String[] colored = BaseFragmentsBuilder.COLORED_PRE_TAGS;
      for (int cfg = 0; cfg < 8; cfg++) {
        for (int qi = 0; qi < queries.size() + BUILT; qi++) {
          for (String field : FIELDS) {
            if (cfg != 0 && !field.equals("tv") && !field.equals("tvsyn")) continue;
            boolean isBuilt = qi >= queries.size();
            String text = isBuilt ? "" : queries.get(qi);
            boolean any = isBuilt ? field.equals("tvsyn") : text.contains("F:");
            if (!any && !field.equals("tv")) continue;
            Query q = isBuilt ? built(qi - queries.size()) : parser.parse(text.replace("F:", field + ":"));
            FastVectorHighlighter fvh;
            FragListBuilder fl = new SimpleFragListBuilder();
            FragmentsBuilder fb = new ScoreOrderFragmentsBuilder();
            int size = 30;
            int max = 3;
            String[] pre = {"<b>"};
            String[] post = {"</b>"};
            Encoder enc = new DefaultEncoder();
            Set<String> matched = null;
            switch (cfg) {
              case 0:
                fvh = new FastVectorHighlighter();
                break;
              case 1:
                fvh = new FastVectorHighlighter(false, false);
                size = 20;
                max = 2;
                break;
              case 2:
                fvh = new FastVectorHighlighter();
                fl = new WeightedFragListBuilder(3);
                SimpleFragmentsBuilder sfb =
                    new SimpleFragmentsBuilder(
                        BaseFragmentsBuilder.COLORED_PRE_TAGS, BaseFragmentsBuilder.COLORED_POST_TAGS);
                sfb.setDiscreteMultiValueHighlighting(true);
                fb = sfb;
                pre = colored;
                post = BaseFragmentsBuilder.COLORED_POST_TAGS;
                size = 25;
                max = 4;
                break;
              case 3:
                fvh = new FastVectorHighlighter(true, false);
                fl = new SingleFragListBuilder();
                fb =
                    new ScoreOrderFragmentsBuilder(
                        new BreakIteratorBoundaryScanner(
                            BreakIterator.getSentenceInstance(Locale.ROOT)));
                max = 1;
                break;
              case 4:
                fvh = new FastVectorHighlighter();
                fb =
                    new ScoreOrderFragmentsBuilder(
                        new SimpleBoundaryScanner(5, new char[] {',', '.', ' '}));
                enc = new SimpleHTMLEncoder();
                matched = new LinkedHashSet<>(List.of("tv", "tvb"));
                break;
              case 5:
                fvh = new FastVectorHighlighter();
                fvh.setPhraseLimit(2);
                fl = new SimpleFragListBuilder(0);
                SimpleFragmentsBuilder sb = new SimpleFragmentsBuilder();
                sb.setMultiValuedSeparator('|');
                fb = sb;
                enc = new SimpleHTMLEncoder();
                pre = new String[] {"["};
                post = new String[] {"]"};
                size = 10;
                max = 5;
                break;
              case 7:
                fvh = new FastVectorHighlighter();
                SimpleFragmentsBuilder dmv = new SimpleFragmentsBuilder();
                dmv.setDiscreteMultiValueHighlighting(true);
                fb = dmv;
                size = 60;
                max = 4;
                break;
              default:
                fvh = new FastVectorHighlighter(false, true);
                fl = new WeightedFragListBuilder();
                size = 40;
                max = 2;
                break;
            }
            FieldQuery fq = fvh.getFieldQuery(q, reader);
            for (int doc = 0; doc < reader.maxDoc(); doc++) {
              rows.append(cfg).append((char) 9).append(qi).append((char) 9).append(field)
                  .append((char) 9).append(doc).append((char) 9);
              try {
                String[] frags;
                if (matched != null) {
                  frags = fvh.getBestFragments(fq, reader, doc, field, matched, size, max, fl, fb, pre, post, enc);
                } else {
                  frags = fvh.getBestFragments(fq, reader, doc, field, size, max, fl, fb, pre, post, enc);
                }
                if (frags == null) {
                  rows.append("N\t");
                } else {
                  rows.append("R\t").append(esc(String.join("\u001f", frags)));
                }
              } catch (Exception e) {
                rows.append("X\t");
              }
              rows.append((char) 10);
            }
          }
        }
      }
    }
    Files.writeString(out.resolve("vhighlight.tsv"), rows.toString(), StandardCharsets.UTF_8);
  }
}
