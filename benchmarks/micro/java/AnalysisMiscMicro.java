import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.classic.ClassicAnalyzer;
import org.apache.lucene.analysis.compound.DictionaryCompoundWordTokenFilter;
import org.apache.lucene.analysis.compound.HyphenationCompoundWordTokenFilter;
import org.apache.lucene.analysis.compound.hyphenation.HyphenationTree;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.miscellaneous.DateRecognizerFilter;
import org.apache.lucene.analysis.miscellaneous.WordDelimiterFilter;
import org.apache.lucene.analysis.miscellaneous.WordDelimiterGraphFilter;
import org.apache.lucene.analysis.synonym.word2vec.Word2VecModel;
import org.apache.lucene.analysis.synonym.word2vec.Word2VecSynonymFilter;
import org.apache.lucene.analysis.synonym.word2vec.Word2VecSynonymProvider;
import org.apache.lucene.analysis.wikipedia.WikipediaTokenizer;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.TermAndVector;
import org.xml.sax.InputSource;

/**
 * M11 part 3's remaining pairs (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_analysis_misc.rs}): the decompounders, the deprecated {@code
 * WordDelimiterFilter}, {@code ClassicAnalyzer}, {@code WikipediaTokenizer}, {@code
 * DateRecognizerFilter} and {@code Word2VecSynonymFilter}, each over a corpus both sides generate
 * from {@link SweepMicro.Rng} (so no input file is shared but the bytes are equal). Units are
 * tokens; output is {@code name\tns_per_token\ttokens}.
 */
public class AnalysisMiscMicro {
  static final String ALPHA = "abcdefgh";

  static long rem(long x, long n) {
    return Long.remainderUnsigned(x, n);
  }

  static String word(SweepMicro.Rng r, int len) {
    StringBuilder w = new StringBuilder();
    for (int k = 0; k < len; k++) w.append(ALPHA.charAt((int) rem(r.next(), ALPHA.length())));
    return w.toString();
  }

  /** The decompounders' dictionary: 300 words of 2 to 6 letters over {@link #ALPHA}. */
  static List<String> compoundDict() {
    SweepMicro.Rng r = new SweepMicro.Rng(0x5EED_C0DE_1234_5678L);
    List<String> dict = new ArrayList<>();
    for (int i = 0; i < 300; i++) dict.add(word(r, 2 + (int) rem(r.next(), 5)));
    return dict;
  }

  /** 200 documents of 100 words of 6 to 25 letters, every third one suffixed {@code -X9y}. */
  static List<String> compoundDocs() {
    SweepMicro.Rng r = new SweepMicro.Rng(0x0DDC_0FFE_E0DD_BA11L);
    List<String> docs = new ArrayList<>();
    for (int d = 0; d < 200; d++) {
      StringBuilder s = new StringBuilder();
      for (int w = 0; w < 100; w++) {
        if (w > 0) s.append(' ');
        s.append(word(r, 6 + (int) rem(r.next(), 20)));
        if (w % 3 == 0) s.append("-X9y");
      }
      docs.add(s.toString());
    }
    return docs;
  }

  /** An FOP hyphenation grammar: the alphabet's classes and 3000 random patterns. */
  static String hyphenationXml() {
    SweepMicro.Rng r = new SweepMicro.Rng(0x4859_5048_4E41_5445L);
    StringBuilder x =
        new StringBuilder("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<hyphenation-info>\n<classes>\n");
    for (char c : ALPHA.toCharArray()) x.append(c).append(Character.toUpperCase(c)).append(' ');
    x.append("\n</classes>\n<patterns>\n");
    for (int i = 0; i < 3000; i++) {
      StringBuilder p = new StringBuilder();
      if (rem(r.next(), 6) == 0) p.append('.');
      int n = 1 + (int) rem(r.next(), 7);
      for (int k = 0; k < n; k++) {
        if (rem(r.next(), 3) == 0) p.append((char) ('0' + rem(r.next(), 10)));
        p.append(ALPHA.charAt((int) rem(r.next(), ALPHA.length())));
      }
      if (rem(r.next(), 3) == 0) p.append((char) ('0' + rem(r.next(), 10)));
      if (rem(r.next(), 6) == 0) p.append('.');
      x.append(p).append(i % 12 == 11 ? '\n' : ' ');
    }
    x.append("\n</patterns>\n</hyphenation-info>\n");
    return x.toString();
  }

  static final String[] WIKI = {
    "[[%s]]", "[[Category:%s]]", "'''%s'''", "''%s''", "[http://example.com/%s %s]", "{{cite %s}}",
    "<ref>%s</ref>", "== %s ==", "%s"
  };

  /** 1000 documents of 60 items of wiki markup around {@code w<base36>} words. */
  static List<String> wikiDocs() {
    SweepMicro.Rng r = new SweepMicro.Rng(0x5749_4B49_5045_4449L);
    List<String> docs = new ArrayList<>();
    for (int d = 0; d < 1000; d++) {
      StringBuilder s = new StringBuilder();
      for (int w = 0; w < 60; w++) {
        if (w > 0) s.append(' ');
        long x = r.next();
        String word = "w" + Long.toString(rem(x, 5000), 36);
        s.append(WIKI[(int) rem(x >>> 32, WIKI.length)].replace("%s", word));
      }
      docs.add(s.toString());
    }
    return docs;
  }

  /** 2000 documents of 50 tokens: ISO dates, truncated dates and words. */
  static List<String> isoDateDocs() {
    SweepMicro.Rng r = new SweepMicro.Rng(0x0DA7_E150_0DA7_E150L);
    List<String> docs = new ArrayList<>();
    for (int d = 0; d < 2000; d++) {
      StringBuilder s = new StringBuilder();
      for (int w = 0; w < 50; w++) {
        if (w > 0) s.append(' ');
        long x = r.next();
        long year = 1900 + rem(x >>> 8, 200), month = 1 + rem(x >>> 16, 12), day = 1 + rem(x >>> 24, 28);
        switch ((int) (x & 3)) {
          case 0 -> s.append(String.format(Locale.ROOT, "%04d-%02d-%02d", year, month, day));
          case 1 -> s.append(String.format(Locale.ROOT, "%04d-%02d", year, month));
          default -> s.append('t').append(Long.toString(rem(x >>> 8, 50000), 36));
        }
      }
      docs.add(s.toString());
    }
    return docs;
  }

  static final String[] MONTHS = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
  };

  /** 20000 single-term documents: {@code MMM d, y} dates and words. */
  static List<String> englishDateDocs() {
    SweepMicro.Rng r = new SweepMicro.Rng(0x0DA7_E0E0_0DA7_E0E0L);
    List<String> docs = new ArrayList<>();
    for (int d = 0; d < 20000; d++) {
      long x = r.next();
      if ((x & 1) == 0) {
        docs.add(MONTHS[(int) rem(x >>> 8, 12)] + " " + (1 + rem(x >>> 16, 28)) + ", " + (1900 + rem(x >>> 24, 200)));
      } else {
        docs.add("t" + Long.toString(rem(x >>> 8, 50000), 36));
      }
    }
    return docs;
  }

  static final int W2V_TERMS = 2000;
  static final int W2V_DIM = 50;

  /** A model of {@code v<base36>} terms with random components in [-1, 1]. */
  static Word2VecModel w2vModel() {
    SweepMicro.Rng r = new SweepMicro.Rng(0x5752_3256_4543_5452L);
    Word2VecModel m = new Word2VecModel(W2V_TERMS, W2V_DIM);
    for (int i = 0; i < W2V_TERMS; i++) {
      float[] v = new float[W2V_DIM];
      for (int k = 0; k < W2V_DIM; k++) v[k] = (rem(r.next(), 2001) - 1000) / 1000f;
      m.addTermAndVector(new TermAndVector(new BytesRef("v" + Long.toString(i, 36)), v));
    }
    return m;
  }

  /** 500 documents of 40 of the model's terms. */
  static List<String> w2vDocs() {
    SweepMicro.Rng r = new SweepMicro.Rng(0x5752_3244_4F43_5321L);
    List<String> docs = new ArrayList<>();
    for (int d = 0; d < 500; d++) {
      StringBuilder s = new StringBuilder();
      for (int w = 0; w < 40; w++) {
        if (w > 0) s.append(' ');
        s.append('v').append(Long.toString(rem(r.next(), W2V_TERMS), 36));
      }
      docs.add(s.toString());
    }
    return docs;
  }

  static Analyzer chain(Supplier<Tokenizer> tokenizer, Function<TokenStream, TokenStream> filters) {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer t = tokenizer.get();
        return new TokenStreamComponents(t, filters.apply(t));
      }
    };
  }

  static void run(String name, Analyzer a, List<String> docs) throws IOException {
    SweepMicro.measure(
        name,
        () -> {
          long tokens = 0;
          for (String text : docs) tokens += SweepMicro.consume(a.tokenStream("body", text));
          return tokens;
        });
    a.close();
  }

  public static void main(String[] args) throws Exception {
    List<String> docs = SweepMicro.analysisDocs();
    List<String> compound = compoundDocs();
    CharArraySet dict = new CharArraySet(compoundDict(), true);
    HyphenationTree tree =
        HyphenationCompoundWordTokenFilter.getHyphenationTree(
            new InputSource(new StringReader(hyphenationXml())));
    int wdf =
        WordDelimiterFilter.GENERATE_WORD_PARTS
            | WordDelimiterFilter.GENERATE_NUMBER_PARTS
            | WordDelimiterFilter.SPLIT_ON_CASE_CHANGE
            | WordDelimiterFilter.SPLIT_ON_NUMERICS
            | WordDelimiterFilter.STEM_ENGLISH_POSSESSIVE;
    int all = 511;
    Word2VecSynonymProvider provider = new Word2VecSynonymProvider(w2vModel());
    run("dict_compound", chain(WhitespaceTokenizer::new, t -> new DictionaryCompoundWordTokenFilter(t, dict)), compound);
    run("hyph_compound", chain(WhitespaceTokenizer::new, t -> new HyphenationCompoundWordTokenFilter(t, tree, dict)), compound);
    run("hyph_compound_nodict", chain(WhitespaceTokenizer::new, t -> new HyphenationCompoundWordTokenFilter(t, tree)), compound);
    run("wdf", chain(WhitespaceTokenizer::new, t -> new WordDelimiterFilter(t, wdf, null)), docs);
    run("wdf_all", chain(WhitespaceTokenizer::new, t -> new WordDelimiterFilter(t, all, null)), compound);
    run("wdgf_all", chain(WhitespaceTokenizer::new, t -> new WordDelimiterGraphFilter(t, all, null)), compound);
    run("classic", new ClassicAnalyzer(), docs);
    run("wikipedia", chain(WikipediaTokenizer::new, t -> t), wikiDocs());
    run("date_iso", chain(WhitespaceTokenizer::new,
        t -> new DateRecognizerFilter(t, new java.text.SimpleDateFormat("yyyy-MM-dd", Locale.ENGLISH))), isoDateDocs());
    run("date_default", chain(KeywordTokenizer::new, DateRecognizerFilter::new), englishDateDocs());
    run("word2vec_synonym", chain(WhitespaceTokenizer::new, t -> new Word2VecSynonymFilter(t, provider, 5, 0f)), w2vDocs());
  }
}
