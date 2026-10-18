import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.search.suggest.analyzing.SuggestStopFilter;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.search.suggest.Lookup;
import org.apache.lucene.search.suggest.Lookup.LookupResult;
import org.apache.lucene.search.suggest.analyzing.AnalyzingSuggester;
import org.apache.lucene.search.suggest.analyzing.FuzzySuggester;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.OutputStreamDataOutput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.CharsRef;

/**
 * M13 T13.3c: {@code AnalyzingSuggester} and {@code FuzzySuggester} over a phrase corpus (stop
 * words, case variants of one analyzed form, a synonym graph, non-ASCII). Writes {@code
 * suggest_analyzing/}: {@code phrases.tsv} (surface, weight, payload), {@code queries.tsv}, and
 * per configuration {@code NAME.tsv} (query, num, results), {@code NAME_reload.tsv} (the same
 * through {@code load()}), {@code NAME.bin} ({@code store()}), {@code NAME.count}, plus {@code
 * errors.tsv}.
 */
public class GenAnalyzingSuggest {
  record Phrase(String surface, long weight, String payload) {}

  static final String[][] CORPUS = {
    {"the ghost of christmas past", "100"},
    {"ghost chrystal", "40"},
    {"Ghost Chrystal", "55"},
    {"ghost", "7"},
    {"fast apple", "30"},
    {"Fast Apple", "90"},
    {"The Fast Apple", "20"},
    {"fast applesauce", "12"},
    {"net", "50"},
    {"nets", "49"},
    {"netflix and chill", "60"},
    {"new york", "80"},
    {"New York", "81"},
    {"new york city", "70"},
    {"newyork", "15"},
    {"new yorker", "33"},
    {"the quick brown fox", "25"},
    {"quick fox", "26"},
    {"quick brown", "27"},
    {"a b c", "5"},
    {"of the", "3"},
    {"the of a", "4"},
    {"café au lait", "45"},
    {"cafe latte", "44"},
    {"日本語 text", "35"},
    {"naïve bayes", "22"},
    {"wifi router", "77"},
    {"wireless network", "66"},
    {"wireless wifi bridge", "11"},
    {"zebra crossing", "1"},
    {"zebra", "0"},
    {"max weight", String.valueOf(Integer.MAX_VALUE)},
  };

  static List<Phrase> phrases(boolean payloads) {
    List<Phrase> out = new ArrayList<>();
    int i = 0;
    for (String[] c : CORPUS) {
      out.add(new Phrase(c[0], Long.parseLong(c[1]), payloads ? "pay" + (i++) : null));
    }
    return out;
  }

  static final class Entries implements InputIterator {
    final List<Phrase> phrases;
    final boolean payloads;
    int next = 0;
    Phrase cur;

    Entries(List<Phrase> phrases, boolean payloads) {
      this.phrases = phrases;
      this.payloads = payloads;
    }

    @Override
    public BytesRef next() {
      if (next >= phrases.size()) return null;
      cur = phrases.get(next++);
      return new BytesRef(cur.surface());
    }

    @Override
    public long weight() {
      return cur.weight();
    }

    @Override
    public BytesRef payload() {
      return payloads ? new BytesRef(cur.payload()) : null;
    }

    @Override
    public boolean hasPayloads() {
      return payloads;
    }

    @Override
    public Set<BytesRef> contexts() {
      return null;
    }

    @Override
    public boolean hasContexts() {
      return false;
    }
  }

  static CharArraySet stops() {
    return new CharArraySet(List.of("the", "of", "a"), false);
  }

  static Analyzer standard() {
    return new StandardAnalyzer(stops());
  }

  /** Tokenizer, lower case, stop words, then a one-way synonym graph. */
  static Analyzer synonyms() throws Exception {
    SynonymMap.Builder b = new SynonymMap.Builder(true);
    b.add(new CharsRef("wifi"), new CharsRef("wireless"), true);
    SynonymMap map = b.build();
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String field) {
        Tokenizer source = new StandardTokenizer();
        TokenStream ts = new LowerCaseFilter(source);
        ts = new StopFilter(ts, stops());
        ts = new SynonymGraphFilter(ts, map, true);
        return new TokenStreamComponents(source, ts);
      }
    };
  }

  /** Tokenizer, lower case, then {@code SuggestStopFilter}: a trailing stop word is kept. */
  static Analyzer suggestStop() {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String field) {
        Tokenizer source = new StandardTokenizer();
        TokenStream ts = new LowerCaseFilter(source);
        ts = new SuggestStopFilter(ts, stops());
        return new TokenStreamComponents(source, ts);
      }
    };
  }

  static final String[] STOP_INPUTS = {
    "", "the", "the ", "of the", "of the ", "ghost of", "ghost of ", "ghost of the christmas", "a b a", "a b a ",
    "THE", "The  fast", "fast of  ", "a", "a a a", "a a a ", "x the y of", "café of", "日本語 the", "the.", "the,"
  };

  static void suggestStopRows(Path out) throws Exception {
    StringBuilder rows = new StringBuilder();
    try (Analyzer a = suggestStop()) {
      for (String in : STOP_INPUTS) {
        rows.append(AnalysisRows.esc(in)).append('\t');
        try (TokenStream ts = a.tokenStream("", in)) {
          CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
          PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
          OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
          KeywordAttribute kw = ts.addAttribute(KeywordAttribute.class);
          ts.reset();
          boolean first = true;
          while (ts.incrementToken()) {
            if (!first) rows.append('|');
            first = false;
            rows.append(AnalysisRows.esc(term.toString())).append(',').append(off.startOffset()).append(',')
                .append(off.endOffset()).append(',').append(inc.getPositionIncrement()).append(',').append(kw.isKeyword());
          }
          ts.end();
          rows.append('\t').append(inc.getPositionIncrement()).append(',').append(off.endOffset());
        }
        rows.append('\n');
      }
    }
    Files.writeString(out.resolve("suggest_stop.tsv"), rows.toString(), StandardCharsets.UTF_8);
  }

  static String hex(byte[] b) {
    StringBuilder s = new StringBuilder();
    for (byte x : b) s.append(String.format("%02x", x & 0xff));
    return s.toString();
  }

  static String results(List<LookupResult> res) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < res.size(); i++) {
      if (i > 0) b.append('\u001f');
      LookupResult r = res.get(i);
      b.append(AnalysisRows.esc(r.key.toString())).append('\u001e').append(r.value);
      b.append('\u001e');
      if (r.payload != null) {
        b.append(hex(java.util.Arrays.copyOfRange(r.payload.bytes, r.payload.offset, r.payload.offset + r.payload.length)));
      }
    }
    return b.toString();
  }

  static String run(java.util.concurrent.Callable<List<LookupResult>> c) {
    try {
      return "R\t" + results(c.call());
    } catch (Exception e) {
      return "X\t" + e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  static List<String> queries() {
    Set<String> q = new LinkedHashSet<>();
    for (String[] c : CORPUS) {
      String p = c[0];
      q.add(p);
      q.add(p + " ");
      String lower = p.toLowerCase(java.util.Locale.ROOT);
      String[] words = lower.split(" ");
      StringBuilder acc = new StringBuilder();
      for (int i = 0; i < words.length; i++) {
        if (i > 0) acc.append(' ');
        acc.append(words[i]);
        q.add(acc.toString());
      }
      for (int n : new int[] {1, 2, 3, 4, 6}) {
        if (n < lower.length()) q.add(lower.substring(0, lower.offsetByCodePoints(0, Math.min(n, lower.codePointCount(0, lower.length())))));
      }
    }
    for (String h : new String[] {"", "ghost chr", "the ghost", "fast a", "net ", "nwe", "newyork", "caf", "cafe",
        "日本", "zzz", "qick fox", "ghos chrystal", "a", "the", "of the", "wirele", "wifi", "netflx", "nw yor",
        "naive", "max", "zebr", "zebra crosing"}) {
      q.add(h);
    }
    return new ArrayList<>(q);
  }

  static void lookups(Path out, String name, Lookup lk, List<String> queries) throws Exception {
    StringBuilder rows = new StringBuilder();
    for (String qy : queries) {
      for (int num : new int[] {1, 3, 10}) {
        rows.append(AnalysisRows.esc(qy)).append('\t').append(num).append('\t');
        rows.append(run(() -> lk.lookup(qy, false, num))).append('\n');
      }
    }
    Files.writeString(out.resolve(name + ".tsv"), rows.toString(), StandardCharsets.UTF_8);
  }

  static void store(Path out, String name, Lookup lk) throws Exception {
    try (OutputStream os = Files.newOutputStream(out.resolve(name + ".bin"))) {
      boolean ok = lk.store(new OutputStreamDataOutput(os));
      Files.writeString(out.resolve(name + ".stored"), ok ? "true\n" : "false\n");
    }
  }

  interface Factory {
    Lookup make() throws Exception;
  }

  static void config(Path out, String name, Factory f, boolean payloads, List<String> queries) throws Exception {
    Lookup lk = f.make();
    lk.build(new Entries(phrases(payloads), payloads));
    lookups(out, name, lk, queries);
    store(out, name, lk);
    Files.writeString(out.resolve(name + ".count"), lk.getCount() + "\n");
    Lookup again = f.make();
    try (var in = Files.newInputStream(out.resolve(name + ".bin"))) {
      again.load(in);
    }
    lookups(out, name + "_reload", again, queries);
  }

  static AnalyzingSuggester an(Analyzer a, int options, int maxSurface, int maxGraph, boolean preservePos) {
    return new AnalyzingSuggester(new ByteBuffersDirectory(), "an", a, a, options, maxSurface, maxGraph, preservePos);
  }

  static FuzzySuggester fz(Analyzer a, int options, int maxEdits, boolean transpositions, int nonFuzzyPrefix,
      int minFuzzyLength, boolean unicode, boolean preservePos) {
    return new FuzzySuggester(new ByteBuffersDirectory(), "fz", a, a, options, 256, -1, preservePos, maxEdits,
        transpositions, nonFuzzyPrefix, minFuzzyLength, unicode);
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("suggest_analyzing");
    Files.createDirectories(out);
    StringBuilder terms = new StringBuilder();
    int i = 0;
    for (Phrase p : phrases(true)) {
      terms.append(AnalysisRows.esc(p.surface())).append('\t').append(p.weight()).append('\t').append(p.payload()).append('\n');
    }
    Files.writeString(out.resolve("phrases.tsv"), terms.toString(), StandardCharsets.UTF_8);
    List<String> queries = queries();
    StringBuilder qs = new StringBuilder();
    for (String q : queries) qs.append(AnalysisRows.esc(q)).append('\n');
    Files.writeString(out.resolve("queries.tsv"), qs.toString(), StandardCharsets.UTF_8);

    int both = AnalyzingSuggester.EXACT_FIRST | AnalyzingSuggester.PRESERVE_SEP;
    config(out, "a_default", () -> new AnalyzingSuggester(new ByteBuffersDirectory(), "an", standard()), false, queries);
    config(out, "a_nosep", () -> an(standard(), AnalyzingSuggester.EXACT_FIRST, 256, -1, false), false, queries);
    config(out, "a_plain", () -> an(standard(), 0, 2, -1, true), false, queries);
    config(out, "a_payload", () -> an(standard(), both, 256, -1, true), true, queries);
    config(out, "a_syn", () -> an(synonyms(), both, 256, -1, true), false, queries);
    config(out, "a_syn_limit", () -> an(synonyms(), both, 256, 1, true), false, queries);
    config(out, "a_suggeststop", () -> new AnalyzingSuggester(new ByteBuffersDirectory(), "an", standard(), suggestStop(), both, 256, -1, true), false, queries);
    suggestStopRows(out);
    config(out, "f_default", () -> new FuzzySuggester(new ByteBuffersDirectory(), "fz", standard()), false, queries);
    config(out, "f_unicode", () -> fz(standard(), both, 2, false, 0, 2, true, true), false, queries);
    config(out, "f_nosep", () -> fz(standard(), AnalyzingSuggester.EXACT_FIRST, 1, true, 2, 3, false, false), false, queries);
    config(out, "f_payload", () -> fz(standard(), both, 1, true, 1, 3, false, true), true, queries);
    config(out, "f_syn", () -> fz(synonyms(), both, 1, true, 1, 3, false, true), false, queries);

    // an empty build stores a count and refuses to save
    AnalyzingSuggester empty = new AnalyzingSuggester(new ByteBuffersDirectory(), "an", standard());
    empty.build(new Entries(new ArrayList<>(), false));
    store(out, "a_empty", empty);
    Files.writeString(out.resolve("a_empty.lookup"), run(() -> empty.lookup("a", false, 3)) + "\n");

    // refusals
    StringBuilder errs = new StringBuilder();
    AnalyzingSuggester ok = new AnalyzingSuggester(new ByteBuffersDirectory(), "an", standard());
    ok.build(new Entries(phrases(false), false));
    errs.append("only_more_popular\t").append(run(() -> ok.lookup("ne", true, 3))).append('\n');
    errs.append("contexts\t").append(run(() -> ok.lookup("ne", new java.util.HashSet<>(List.of(new BytesRef("c"))), false, 3))).append('\n');
    errs.append("hole_key\t").append(run(() -> ok.lookup("a\u001eb", false, 3))).append('\n');
    errs.append("sep_key\t").append(run(() -> ok.lookup("a\u001fb", false, 3))).append('\n');
    for (int options : new int[] {4, -1, 3}) {
      errs.append("options_").append(options).append('\t').append(ctor(() -> an(standard(), options, 256, -1, true))).append('\n');
    }
    for (int surfaces : new int[] {0, 257, 256, 1}) {
      errs.append("surfaces_").append(surfaces).append('\t').append(ctor(() -> an(standard(), both, surfaces, -1, true))).append('\n');
    }
    for (int graph : new int[] {0, -2, -1, 1}) {
      errs.append("graph_").append(graph).append('\t').append(ctor(() -> an(standard(), both, 256, graph, true))).append('\n');
    }
    errs.append("fuzzy_edits_3\t").append(ctor(() -> fz(standard(), both, 3, true, 1, 3, false, true))).append('\n');
    errs.append("fuzzy_edits_-1\t").append(ctor(() -> fz(standard(), both, -1, true, 1, 3, false, true))).append('\n');
    errs.append("fuzzy_prefix_-1\t").append(ctor(() -> fz(standard(), both, 1, true, -1, 3, false, true))).append('\n');
    errs.append("fuzzy_min_-1\t").append(ctor(() -> fz(standard(), both, 1, true, 1, -1, false, true))).append('\n');
    errs.append("get\t").append(getError(ok)).append('\n');
    errs.append("negative_weight\t").append(buildError(List.of(new Phrase("a b", -1, null)), false)).append('\n');
    errs.append("big_weight\t").append(buildError(List.of(new Phrase("a b", (long) Integer.MAX_VALUE + 1, null)), false)).append('\n');
    errs.append("sep_in_surface\t").append(buildError(List.of(new Phrase("a\u001fb", 1, "p")), true)).append('\n');
    errs.append("sep_in_surface_no_payload\t").append(buildError(List.of(new Phrase("a\u001fb", 1, null)), false)).append('\n');
    Files.writeString(out.resolve("errors.tsv"), errs.toString(), StandardCharsets.UTF_8);
  }

  static String ctor(Factory f) {
    try {
      f.make();
      return "ok";
    } catch (Exception e) {
      return e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  static String getError(AnalyzingSuggester s) {
    try {
      return "R\t" + s.get("net");
    } catch (Exception e) {
      return e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  static String buildError(List<Phrase> phrases, boolean payloads) {
    try {
      AnalyzingSuggester s = new AnalyzingSuggester(new ByteBuffersDirectory(), "an", standard());
      s.build(new Entries(phrases, payloads));
      return "built\t" + run(() -> s.lookup("a", false, 3));
    } catch (Exception e) {
      return e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }
}
