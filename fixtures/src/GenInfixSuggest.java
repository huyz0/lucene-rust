import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.search.suggest.Lookup.LookupResult;
import org.apache.lucene.search.suggest.analyzing.AnalyzingInfixSuggester;
import org.apache.lucene.search.suggest.analyzing.BlendedInfixSuggester;
import org.apache.lucene.search.suggest.analyzing.BlendedInfixSuggester.BlenderType;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.3c: {@code AnalyzingInfixSuggester} and {@code BlendedInfixSuggester} over a title
 * corpus with weights, payloads and contexts. Writes {@code suggest_infix/}: {@code corpus.tsv},
 * {@code queries.tsv}, per configuration {@code NAME.tsv} (variant, query, num, results) and {@code
 * NAME.count}, {@code nrt.tsv} (lookups between {@code add}/{@code update}/{@code refresh}) and
 * {@code errors.tsv}.
 */
public class GenInfixSuggest {
  record Item(String text, long weight, String payload, List<String> contexts) {}

  static final Item[] CORPUS = {
    new Item("Lend me your ears", 100, "p0", List.of("red")),
    new Item("Mend and lend", 90, "p1", List.of("red", "blue")),
    new Item("The lender of last resort", 80, "p2", List.of("blue")),
    new Item("Pretend to lend", 70, "p3", List.of()),
    new Item("Fast apple pie", 60, "p4", List.of("green")),
    new Item("Apple fast food", 60, "p5", List.of("green", "red")),
    new Item("Pineapple and apple juice", 55, "p6", List.of("blue")),
    new Item("Big Apple", 55, "p7", List.of("red")),
    new Item("apple", 5, "p8", List.of("green")),
    new Item("Apples", 0, "p9", List.of("blue")),
    new Item("quick brown fox", 40, "p10", List.of("red")),
    new Item("The quick brown dog", 39, "p11", List.of("red", "green")),
    new Item("brown quick fox", 38, "p12", List.of("blue")),
    new Item("lazy dog sleeps", 37, "p13", List.of("green")),
    new Item("a b c d e f", 36, "p14", List.of("red")),
    new Item("f e d c b a", 36, "p15", List.of("red")),
    new Item("café au lait", 35, "p16", List.of("blue")),
    new Item("Cafeteria menu", 34, "p17", List.of("green")),
    new Item("日本語 text here", 33, "p18", List.of("red")),
    new Item("negative weight", -7, "p19", List.of("red")),
    new Item("tiny weight", 3, "p20", List.of("blue")),
    new Item("tiny weights matter", -3, "p21", List.of("blue")),
    new Item("Lend me your ears", 100, "p22", List.of("blue")),
    new Item("repeat repeat repeat", 20, "p23", List.of("green")),
    new Item("Big Apple Pie", 55, "p24", List.of("green")),
    new Item("of the of", 19, "p25", List.of()),
  };

  static final class Entries implements InputIterator {
    final List<Item> items;
    final boolean payloads;
    final boolean contexts;
    int next = -1;

    Entries(List<Item> items, boolean payloads, boolean contexts) {
      this.items = items;
      this.payloads = payloads;
      this.contexts = contexts;
    }

    @Override
    public BytesRef next() {
      return ++next < items.size() ? new BytesRef(items.get(next).text()) : null;
    }

    @Override
    public long weight() {
      return items.get(next).weight();
    }

    @Override
    public BytesRef payload() {
      return payloads ? new BytesRef(items.get(next).payload()) : null;
    }

    @Override
    public boolean hasPayloads() {
      return payloads;
    }

    @Override
    public Set<BytesRef> contexts() {
      if (!contexts) return null;
      Set<BytesRef> out = new TreeSet<>();
      for (String c : items.get(next).contexts()) out.add(new BytesRef(c));
      return out;
    }

    @Override
    public boolean hasContexts() {
      return contexts;
    }
  }

  static Analyzer standard() {
    return new StandardAnalyzer(CharArraySet.EMPTY_SET);
  }

  static Analyzer stopping() {
    return new StandardAnalyzer(new CharArraySet(List.of("the", "of", "a"), false));
  }

  static String hex(BytesRef b) {
    StringBuilder s = new StringBuilder();
    for (int i = 0; i < b.length; i++) s.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    return s.toString();
  }

  static String results(List<LookupResult> res) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < res.size(); i++) {
      if (i > 0) b.append('\u001f');
      LookupResult r = res.get(i);
      b.append(AnalysisRows.esc(r.key.toString())).append('\u001e');
      b.append(r.highlightKey == null ? "~null" : AnalysisRows.esc(r.highlightKey.toString())).append('\u001e');
      b.append(r.value).append('\u001e');
      b.append(r.payload == null ? "~null" : hex(r.payload)).append('\u001e');
      if (r.contexts == null) {
        b.append("~null");
      } else {
        TreeSet<String> hexes = new TreeSet<>();
        for (BytesRef c : r.contexts) hexes.add(hex(c));
        b.append(String.join(",", hexes));
      }
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
    for (Item it : CORPUS) {
      String[] words = it.text().split(" ");
      StringBuilder acc = new StringBuilder();
      for (int i = 0; i < words.length; i++) {
        if (i > 0) acc.append(' ');
        acc.append(words[i]);
        q.add(acc.toString());
        q.add(acc + " ");
        q.add(words[i]);
        q.add(words[i].toLowerCase(java.util.Locale.ROOT) + " ");
        if (words[i].length() > 1) q.add(words[i].substring(0, words[i].length() - 1));
        q.add(words[i].substring(0, 1));
      }
    }
    for (String h : new String[] {"", " ", "app", "apple ", "the", "the ", "of", "lend", "len", "l", "fast ap",
        "big app", "zzz", "zzz ", "!!!", "a b", "a b ", "c a", "b a", "tiny", "tiny w", "REPEAT", "repeat repeat",
        "pineapple", "pine", "日本", "日本語 t", "café", "caf", "me your", "me your ear", "ears lend", "lend ear"}) {
      q.add(h);
    }
    return new ArrayList<>(q);
  }

  static final Set<BytesRef> RED = Set.of(new BytesRef("red"));
  static final Set<BytesRef> RED_BLUE = Set.of(new BytesRef("red"), new BytesRef("blue"));
  static final Set<BytesRef> GREEN = Set.of(new BytesRef("green"));
  static final Set<BytesRef> NONE = Set.of(new BytesRef("none"));

  static Map<BytesRef, Occur> info(Object... pairs) {
    Map<BytesRef, Occur> m = new TreeMap<>();
    for (int i = 0; i < pairs.length; i += 2) m.put(new BytesRef((String) pairs[i]), (Occur) pairs[i + 1]);
    return m;
  }

  interface Searcher {
    List<LookupResult> lookup(String key, Set<BytesRef> contexts, int num, boolean all, boolean hl) throws Exception;

    List<LookupResult> lookupInfo(String key, Map<BytesRef, Occur> info, int num, boolean all, boolean hl) throws Exception;

    List<LookupResult> lookupDefault(String key, Set<BytesRef> contexts, int num) throws Exception;
  }

  static void lookups(Path out, String name, Searcher s, List<String> queries) throws Exception {
    StringBuilder rows = new StringBuilder();
    for (String qy : queries) {
      for (int num : new int[] {1, 3, 10}) {
        String head = AnalysisRows.esc(qy) + "\t" + num + "\t";
        rows.append("default\t").append(head).append(run(() -> s.lookupDefault(qy, null, num))).append('\n');
        rows.append("any\t").append(head).append(run(() -> s.lookup(qy, null, num, false, true))).append('\n');
        rows.append("all_nohl\t").append(head).append(run(() -> s.lookup(qy, null, num, true, false))).append('\n');
        rows.append("red\t").append(head).append(run(() -> s.lookupDefault(qy, RED, num))).append('\n');
        rows.append("red_blue_any\t").append(head).append(run(() -> s.lookup(qy, RED_BLUE, num, false, true))).append('\n');
        rows.append("green_all\t").append(head).append(run(() -> s.lookup(qy, GREEN, num, true, false))).append('\n');
        rows.append("none\t").append(head).append(run(() -> s.lookup(qy, NONE, num, true, true))).append('\n');
        rows.append("must_not_blue\t").append(head)
            .append(run(() -> s.lookupInfo(qy, info("blue", Occur.MUST_NOT), num, true, true))).append('\n');
        rows.append("must_not_blue_any\t").append(head)
            .append(run(() -> s.lookupInfo(qy, info("blue", Occur.MUST_NOT), num, false, true))).append('\n');
        rows.append("red_must_blue_not\t").append(head)
            .append(run(() -> s.lookupInfo(qy, info("red", Occur.MUST, "blue", Occur.MUST_NOT), num, true, false))).append('\n');
        rows.append("red_should_green_should_any\t").append(head)
            .append(run(() -> s.lookupInfo(qy, info("red", Occur.SHOULD, "green", Occur.SHOULD), num, false, true))).append('\n');
        rows.append("empty_info\t").append(head)
            .append(run(() -> s.lookupInfo(qy, info(), num, true, true))).append('\n');
      }
    }
    Files.writeString(out.resolve(name + ".tsv"), rows.toString(), StandardCharsets.UTF_8);
  }

  static Searcher of(AnalyzingInfixSuggester a) {
    return new Searcher() {
      public List<LookupResult> lookup(String k, Set<BytesRef> c, int n, boolean all, boolean hl) throws Exception {
        return a.lookup(k, c, n, all, hl);
      }

      public List<LookupResult> lookupInfo(String k, Map<BytesRef, Occur> i, int n, boolean all, boolean hl) throws Exception {
        return a.lookup(k, i, n, all, hl);
      }

      public List<LookupResult> lookupDefault(String k, Set<BytesRef> c, int n) throws Exception {
        return a.lookup(k, c, false, n);
      }
    };
  }

  interface Factory {
    AnalyzingInfixSuggester make() throws Exception;
  }

  static void config(Path out, String name, Factory f, List<String> queries) throws Exception {
    AnalyzingInfixSuggester lk = f.make();
    lk.build(new Entries(List.of(CORPUS), true, true));
    lookups(out, name, of(lk), queries);
    Files.writeString(out.resolve(name + ".count"), lk.getCount() + "\n");
    lk.close();
  }

  static AnalyzingInfixSuggester infix(Analyzer index, Analyzer query, int minPrefix, boolean all, boolean hl) throws Exception {
    return new AnalyzingInfixSuggester(new ByteBuffersDirectory(), index, query, minPrefix, false, all, hl);
  }

  static AnalyzingInfixSuggester blended(BlenderType t, int factor, Double exponent, int minPrefix) throws Exception {
    return new BlendedInfixSuggester(
        new ByteBuffersDirectory(), standard(), standard(), minPrefix, t, factor, exponent, false, true, true);
  }

  static String call(Factory f, java.util.concurrent.Callable<Object> c) {
    try {
      return "R\t" + c.call();
    } catch (Exception e) {
      return "X\t" + e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("suggest_infix");
    Files.createDirectories(out);
    StringBuilder corpus = new StringBuilder();
    for (Item it : CORPUS) {
      corpus.append(AnalysisRows.esc(it.text())).append('\t').append(it.weight()).append('\t').append(it.payload())
          .append('\t').append(String.join(",", it.contexts())).append('\n');
    }
    Files.writeString(out.resolve("corpus.tsv"), corpus.toString(), StandardCharsets.UTF_8);
    List<String> queries = queries();
    StringBuilder qs = new StringBuilder();
    for (String q : queries) qs.append(AnalysisRows.esc(q)).append('\n');
    Files.writeString(out.resolve("queries.tsv"), qs.toString(), StandardCharsets.UTF_8);

    config(out, "infix_default", () -> new AnalyzingInfixSuggester(new ByteBuffersDirectory(), standard()), queries);
    config(out, "infix_prefix0", () -> infix(standard(), standard(), 0, true, true), queries);
    config(out, "infix_prefix1", () -> infix(standard(), standard(), 1, true, true), queries);
    config(out, "infix_prefix2", () -> infix(standard(), standard(), 2, true, true), queries);
    config(out, "infix_prefix9", () -> infix(standard(), standard(), 9, true, true), queries);
    config(out, "infix_any_nohl", () -> infix(standard(), standard(), 3, false, false), queries);
    config(out, "infix_stop", () -> infix(stopping(), stopping(), 4, true, true), queries);
    config(out, "infix_mixed", () -> infix(standard(), stopping(), 2, true, true), queries);
    config(out, "blend_linear", () -> blended(BlenderType.POSITION_LINEAR, 10, null, 4), queries);
    config(out, "blend_reciprocal", () -> blended(BlenderType.POSITION_RECIPROCAL, 10, null, 4), queries);
    config(out, "blend_exp", () -> blended(BlenderType.POSITION_EXPONENTIAL_RECIPROCAL, 10, null, 4), queries);
    config(out, "blend_exp15", () -> blended(BlenderType.POSITION_EXPONENTIAL_RECIPROCAL, 3, 1.5, 2), queries);
    config(out, "blend_custom", () -> blended(BlenderType.CUSTOM, 5, null, 4), queries);
    config(out, "blend_factor1", () -> blended(BlenderType.POSITION_LINEAR, 1, null, 4), queries);

    // add/update/refresh between lookups
    StringBuilder nrt = new StringBuilder();
    AnalyzingInfixSuggester s = new AnalyzingInfixSuggester(new ByteBuffersDirectory(), standard());
    nrt.append("added_before_build\t").append(call(null, () -> {
      s.add(new BytesRef("early bird"), Set.of(new BytesRef("red")), 5, new BytesRef("e"));
      return s.getCount();
    })).append('\n');
    nrt.append("lookup_unrefreshed\t").append(run(() -> s.lookup("early", null, false, 5))).append('\n');
    nrt.append("refresh\t").append(call(null, () -> {
      s.refresh();
      return s.getCount();
    })).append('\n');
    nrt.append("lookup_refreshed\t").append(run(() -> s.lookup("early", null, false, 5))).append('\n');
    nrt.append("add_no_contexts\t").append(call(null, () -> {
      s.add(new BytesRef("early riser"), null, 9, new BytesRef("r"));
      s.add(new BytesRef("early start"), null, 9, null);
      s.refresh();
      return s.getCount();
    })).append('\n');
    nrt.append("lookup_second_segment\t").append(run(() -> s.lookup("early", null, false, 5))).append('\n');
    nrt.append("update\t").append(call(null, () -> {
      s.update(new BytesRef("early bird"), Set.of(new BytesRef("blue")), 50, new BytesRef("e2"));
      s.refresh();
      return s.getCount();
    })).append('\n');
    nrt.append("lookup_updated\t").append(run(() -> s.lookup("early", null, false, 5))).append('\n');
    nrt.append("lookup_updated_blue\t").append(run(() -> s.lookup("early", Set.of(new BytesRef("blue")), false, 5))).append('\n');
    nrt.append("commit\t").append(call(null, () -> {
      s.commit();
      return "ok";
    })).append('\n');
    s.build(new Entries(List.of(CORPUS), true, true));
    nrt.append("rebuilt_count\t").append(s.getCount()).append('\n');
    nrt.append("rebuilt_lookup\t").append(run(() -> s.lookup("early", null, false, 5))).append('\n');
    nrt.append("rebuilt_lookup_apple\t").append(run(() -> s.lookup("app", null, false, 5))).append('\n');
    nrt.append("commit_after_close_on_build\t").append(call(null, () -> {
      s.commit();
      return "ok";
    })).append('\n');
    nrt.append("refresh_after_build\t").append(call(null, () -> {
      s.refresh();
      return s.getCount();
    })).append('\n');
    s.close();
    nrt.append("closed_count\t").append(s.getCount()).append('\n');
    Files.writeString(out.resolve("nrt.tsv"), nrt.toString(), StandardCharsets.UTF_8);

    // an open writer after build
    AnalyzingInfixSuggester open = new AnalyzingInfixSuggester(new ByteBuffersDirectory(), standard(), standard(), 4, true, true, true, false);
    open.build(new Entries(List.of(CORPUS).subList(0, 5), true, true));
    StringBuilder keep = new StringBuilder();
    keep.append("count\t").append(open.getCount()).append('\n');
    open.add(new BytesRef("lending library"), null, 1000, new BytesRef("lib"));
    keep.append("before_refresh\t").append(run(() -> open.lookup("lend", null, false, 3))).append('\n');
    open.refresh();
    keep.append("after_refresh\t").append(run(() -> open.lookup("lend", null, false, 3))).append('\n');
    open.commit();
    keep.append("commit\tok\n");
    Files.writeString(out.resolve("open_writer.tsv"), keep.toString(), StandardCharsets.UTF_8);
    open.close();

    // the payloads doc-values field outlives the document that brought it,
    // until its whole segment is gone
    StringBuilder pay = new StringBuilder();
    AnalyzingInfixSuggester p = new AnalyzingInfixSuggester(new ByteBuffersDirectory(), standard());
    p.add(new BytesRef("xray one"), null, 3, new BytesRef("px"));
    p.add(new BytesRef("yankee one"), null, 2, null);
    p.refresh();
    pay.append("mixed\t").append(run(() -> p.lookup("one", null, false, 5))).append('\n');
    p.update(new BytesRef("xray one"), null, 3, null);
    p.refresh();
    pay.append("payload_deleted\t").append(run(() -> p.lookup("one", null, false, 5))).append('\n');
    p.update(new BytesRef("yankee one"), null, 2, null);
    p.refresh();
    pay.append("first_segment_gone\t").append(run(() -> p.lookup("one", null, false, 5))).append('\n');
    p.close();
    Files.writeString(out.resolve("payloads.tsv"), pay.toString(), StandardCharsets.UTF_8);

    StringBuilder errs = new StringBuilder();
    AnalyzingInfixSuggester unbuilt = new AnalyzingInfixSuggester(new ByteBuffersDirectory(), standard());
    errs.append("unbuilt_lookup\t").append(run(() -> unbuilt.lookup("a", null, false, 3))).append('\n');
    errs.append("unbuilt_refresh\t").append(call(null, () -> {
      unbuilt.refresh();
      return "ok";
    })).append('\n');
    errs.append("unbuilt_commit\t").append(call(null, () -> {
      unbuilt.commit();
      return "ok";
    })).append('\n');
    errs.append("unbuilt_count\t").append(unbuilt.getCount()).append('\n');
    errs.append("negative_min_prefix\t").append(call(null, () -> infix(standard(), standard(), -1, true, true) != null ? "ok" : "")).append('\n');
    AnalyzingInfixSuggester ok = new AnalyzingInfixSuggester(new ByteBuffersDirectory(), standard());
    ok.build(new Entries(List.of(CORPUS), true, true));
    errs.append("num_0\t").append(run(() -> ok.lookup("apple", null, false, 0))).append('\n');
    errs.append("num_negative\t").append(run(() -> ok.lookup("apple", null, false, -1))).append('\n');
    errs.append("store\t").append(call(null, () -> ok.store(new org.apache.lucene.store.OutputStreamDataOutput(new java.io.ByteArrayOutputStream())))).append('\n');
    errs.append("load\t").append(call(null, () -> ok.load(new java.io.ByteArrayInputStream(new byte[0])))).append('\n');
    BlendedInfixSuggester b0 = (BlendedInfixSuggester) blended(BlenderType.POSITION_LINEAR, 10, null, 4);
    b0.build(new Entries(List.of(CORPUS), true, true));
    errs.append("blended_num_0\t").append(run(() -> b0.lookup("apple", null, false, 0))).append('\n');
    errs.append("count_after_build\t").append(ok.getCount()).append('\n');
    // payloads/contexts absent
    AnalyzingInfixSuggester bare = new AnalyzingInfixSuggester(new ByteBuffersDirectory(), standard());
    bare.build(new Entries(List.of(CORPUS), false, false));
    errs.append("bare\t").append(run(() -> bare.lookup("apple", null, false, 3))).append('\n');
    errs.append("bare_ctx\t").append(run(() -> bare.lookup("apple", RED, false, 3))).append('\n');
    Files.writeString(out.resolve("errors.tsv"), errs.toString(), StandardCharsets.UTF_8);
  }
}
