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
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.search.suggest.Lookup;
import org.apache.lucene.search.suggest.Lookup.LookupResult;
import org.apache.lucene.search.suggest.analyzing.FreeTextSuggester;
import org.apache.lucene.store.OutputStreamDataOutput;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.3c: {@code FreeTextSuggester} over a small free-text corpus. Writes {@code
 * suggest_freetext/}: {@code corpus.tsv}, {@code queries.tsv}, and per configuration {@code
 * NAME.tsv} (query, num, results), {@code NAME_reload.tsv} (the same through {@code load()}),
 * {@code NAME.bin} ({@code store()}), {@code NAME.count}, plus {@code errors.tsv}.
 */
public class GenFreeTextSuggest {
  static final String[] CORPUS = {
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox runs away",
    "the quick red fox jumps high",
    "a quick brown dog barks at the fox",
    "the lazy dog sleeps all day",
    "the lazy cat sleeps all night",
    "purple mushroom soup is good",
    "purple mushroom soup is hot",
    "purple mushrooms grow in the dark",
    "i love purple mushroom soup",
    "to be or not to be that is the question",
    "to be is to do and to do is to be",
    "Quick quick QUICK brown",
    "café au lait is a quick breakfast",
    "日本語 text and more text",
    "new york new york new jersey",
    "of the of the of the",
  };

  static final class Entries implements InputIterator {
    final List<String> rows;
    final boolean payloads;
    final boolean contexts;
    int next = 0;

    Entries(List<String> rows, boolean payloads, boolean contexts) {
      this.rows = rows;
      this.payloads = payloads;
      this.contexts = contexts;
    }

    @Override
    public BytesRef next() {
      return next < rows.size() ? new BytesRef(rows.get(next++)) : null;
    }

    @Override
    public long weight() {
      return 1;
    }

    @Override
    public BytesRef payload() {
      return payloads ? new BytesRef("p") : null;
    }

    @Override
    public boolean hasPayloads() {
      return payloads;
    }

    @Override
    public Set<BytesRef> contexts() {
      return contexts ? Set.of(new BytesRef("c")) : null;
    }

    @Override
    public boolean hasContexts() {
      return contexts;
    }
  }

  static Analyzer plain() {
    return new StandardAnalyzer(CharArraySet.EMPTY_SET);
  }

  static Analyzer stopping() {
    return new StandardAnalyzer(new CharArraySet(List.of("the", "of", "a", "is", "to"), false));
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
    }
    return b.toString();
  }

  interface Call {
    List<LookupResult> call() throws Exception;
  }

  static String run(Call c) {
    try {
      return "R\t" + results(c.call());
    } catch (Exception e) {
      return "X\t" + e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  static List<String> queries() {
    Set<String> q = new LinkedHashSet<>();
    for (String s : CORPUS) {
      String[] words = s.split(" ");
      StringBuilder acc = new StringBuilder();
      for (int i = 0; i < words.length; i++) {
        if (i > 0) acc.append(' ');
        acc.append(words[i]);
        q.add(acc.toString());
        q.add(acc + " ");
        String w = words[i];
        if (w.length() > 1) {
          q.add(acc.substring(0, acc.length() - 1));
        }
      }
    }
    for (String h : new String[] {"", " ", "the", "the ", "of", "of the", "of the ", "zzz", "zzz ", "!!!", "purple mush",
        "purple mushroom ", "to be ", "to ", "quick brown fox ", "the quick brown fox jumps over the lazy",
        "a b c d", "NEW YORK", "new ", "日本", "日本語 ", "unseen words here", "the unseen", "unseen the",
        "mushroom", "mushroom s", "x the", "brown the ", "fox  ", "  fox"}) {
      q.add(h);
    }
    return new ArrayList<>(q);
  }

  static void lookups(Path out, String name, FreeTextSuggester lk, List<String> queries) throws Exception {
    StringBuilder rows = new StringBuilder();
    for (String qy : queries) {
      for (int num : new int[] {1, 3, 10}) {
        rows.append(AnalysisRows.esc(qy)).append('\t').append(num).append('\t');
        rows.append(run(() -> lk.lookup(qy, null, false, num))).append('\n');
      }
    }
    Files.writeString(out.resolve(name + ".tsv"), rows.toString(), StandardCharsets.UTF_8);
  }

  interface Factory {
    FreeTextSuggester make() throws Exception;
  }

  static void config(Path out, String name, Factory f, List<String> queries) throws Exception {
    FreeTextSuggester lk = f.make();
    lk.build(new Entries(List.of(CORPUS), false, false));
    lookups(out, name, lk, queries);
    try (OutputStream os = Files.newOutputStream(out.resolve(name + ".bin"))) {
      lk.store(new OutputStreamDataOutput(os));
    }
    Files.writeString(out.resolve(name + ".count"), lk.getCount() + "\n");
    FreeTextSuggester again = f.make();
    try (var in = Files.newInputStream(out.resolve(name + ".bin"))) {
      again.load(in);
    }
    lookups(out, name + "_reload", again, queries);
  }

  static String ctor(Factory f) {
    try {
      f.make();
      return "ok";
    } catch (Exception e) {
      return e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  static String build(Factory f, List<String> rows, boolean payloads, boolean contexts) {
    try {
      FreeTextSuggester s = f.make();
      s.build(new Entries(rows, payloads, contexts));
      return "built\t" + s.getCount() + "\t" + run(() -> s.lookup("a", null, false, 3));
    } catch (Exception e) {
      return e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("suggest_freetext");
    Files.createDirectories(out);
    StringBuilder corpus = new StringBuilder();
    for (String s : CORPUS) corpus.append(AnalysisRows.esc(s)).append('\n');
    Files.writeString(out.resolve("corpus.tsv"), corpus.toString(), StandardCharsets.UTF_8);
    List<String> queries = queries();
    StringBuilder qs = new StringBuilder();
    for (String q : queries) qs.append(AnalysisRows.esc(q)).append('\n');
    Files.writeString(out.resolve("queries.tsv"), qs.toString(), StandardCharsets.UTF_8);

    config(out, "bigram", () -> new FreeTextSuggester(plain()), queries);
    config(out, "unigram", () -> new FreeTextSuggester(plain(), plain(), 1), queries);
    config(out, "trigram", () -> new FreeTextSuggester(plain(), plain(), 3), queries);
    config(out, "fourgram", () -> new FreeTextSuggester(plain(), plain(), 4), queries);
    config(out, "stop_bigram", () -> new FreeTextSuggester(stopping(), stopping(), 2), queries);
    config(out, "stop_trigram", () -> new FreeTextSuggester(stopping(), stopping(), 3), queries);
    config(out, "mixed", () -> new FreeTextSuggester(plain(), stopping(), 3, (byte) '|'), queries);

    StringBuilder errs = new StringBuilder();
    errs.append("grams_0\t").append(ctor(() -> new FreeTextSuggester(plain(), plain(), 0))).append('\n');
    errs.append("grams_-1\t").append(ctor(() -> new FreeTextSuggester(plain(), plain(), -1))).append('\n');
    errs.append("separator_high\t").append(ctor(() -> new FreeTextSuggester(plain(), plain(), 2, (byte) 0x80))).append('\n');
    errs.append("separator_nul\t").append(ctor(() -> new FreeTextSuggester(plain(), plain(), 2, (byte) 0))).append('\n');
    errs.append("payloads\t").append(build(() -> new FreeTextSuggester(plain()), List.of("a b"), true, false)).append('\n');
    errs.append("contexts\t").append(build(() -> new FreeTextSuggester(plain()), List.of("a b"), false, true)).append('\n');
    errs.append("no_input\t").append(build(() -> new FreeTextSuggester(plain()), List.of(), false, false)).append('\n');
    errs.append("no_tokens\t").append(build(() -> new FreeTextSuggester(plain()), List.of("!!!", ""), false, false)).append('\n');
    errs.append("single_token\t").append(build(() -> new FreeTextSuggester(plain()), List.of("a"), false, false)).append('\n');
    Factory ws = () -> new FreeTextSuggester(new KeywordAnalyzer(), new KeywordAnalyzer(), 2);
    errs.append("separator_in_token\t").append(build(ws, List.of("a\u001eb c"), false, false)).append('\n');
    errs.append("separator_in_tokens\t").append(build(ws, List.of("a\u001eb\u001ec d"), false, false)).append('\n');
    FreeTextSuggester unbuilt = new FreeTextSuggester(plain());
    errs.append("unbuilt\t").append(run(() -> unbuilt.lookup("a", null, false, 3))).append('\n');
    errs.append("unbuilt_count\t").append(unbuilt.getCount()).append('\n');
    FreeTextSuggester ok = new FreeTextSuggester(plain());
    ok.build(new Entries(List.of(CORPUS), false, false));
    errs.append("lookup_contexts\t").append(run(() -> ok.lookup("a", Set.of(new BytesRef("c")), false, 3))).append('\n');
    errs.append("get\t").append(getError(ok)).append('\n');
    FreeTextSuggester wsBuilt = ws.make();
    wsBuilt.build(new Entries(List.of("a b c"), false, false));
    errs.append("separator_in_query\t").append(run(() -> wsBuilt.lookup("x\u001ey z", null, false, 3))).append('\n');
    errs.append("lookup_zero\t").append(run(() -> ok.lookup("the", null, false, 0))).append('\n');
    // loading a model built with other settings
    Path bin = out.resolve("bigram.bin");
    errs.append("load_grams\t").append(loadError(() -> new FreeTextSuggester(plain(), plain(), 3), bin)).append('\n');
    errs.append("load_separator\t").append(loadError(() -> new FreeTextSuggester(plain(), plain(), 2, (byte) '|'), bin)).append('\n');
    Files.writeString(out.resolve("errors.tsv"), errs.toString(), StandardCharsets.UTF_8);
  }

  static String getError(Lookup s) {
    try {
      return "R\t" + ((FreeTextSuggester) s).get("net");
    } catch (Exception e) {
      return e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  static String loadError(Factory f, Path bin) {
    try (var in = Files.newInputStream(bin)) {
      f.make().load(in);
      return "loaded";
    } catch (Exception e) {
      return e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }
}
