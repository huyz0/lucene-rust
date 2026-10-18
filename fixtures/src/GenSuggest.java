import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.search.suggest.Lookup;
import org.apache.lucene.search.suggest.Lookup.LookupResult;
import org.apache.lucene.search.suggest.tst.TSTLookup;
import org.apache.lucene.store.OutputStreamDataOutput;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.3b: the suggesters of {@code lucene-suggest}, one fixture directory {@code suggest/}.
 * {@code terms.tsv} is the build input ({@code term TAB weight}, input order); {@code tst.tsv} is
 * {@code TSTLookup}'s answer to every short prefix of it, with and without {@code onlyMorePopular};
 * {@code tst.bin} is {@code store()}'s bytes; {@code tst_get.tsv} is {@code get()}.
 */
public class GenSuggest {
  record Entry(String term, long weight) {}

  /** The weighted-term stream every suggester here is built from. */
  static final class Entries implements InputIterator {
    private final List<Entry> entries;
    private int next = 0;
    private Entry cur;

    Entries(List<Entry> entries) {
      this.entries = entries;
    }

    @Override
    public BytesRef next() {
      if (next >= entries.size()) return null;
      cur = entries.get(next++);
      return new BytesRef(cur.term());
    }

    @Override
    public long weight() {
      return cur.weight();
    }

    @Override
    public BytesRef payload() {
      return null;
    }

    @Override
    public boolean hasPayloads() {
      return false;
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

  static List<Entry> corpus() {
    Random r = new Random(1305);
    String[] syl = {"ca", "re", "ton", "pa", "li", "sa", "mo", "ne", "du", "fi"};
    List<Entry> out = new ArrayList<>();
    // Words of one to four syllables; a few repeat, with different weights.
    for (int i = 0; i < 300; i++) {
      StringBuilder b = new StringBuilder();
      int n = 1 + r.nextInt(4);
      for (int j = 0; j < n; j++) b.append(syl[r.nextInt(syl.length)]);
      out.add(new Entry(b.toString(), r.nextInt(1000)));
    }
    String[] special = {
      "", "a", "ab", "abc", "café", "cafe", "naïve", "日本", "日本語",
      "日本語学校", "😀", "😀smile", "�", "",
      "x", "￮x", "x😀y", "Zed", "zed", "ZED"
    };
    for (String s : special) out.add(new Entry(s, r.nextInt(1000)));
    // The same key twice, the second heavier then lighter than the first.
    out.add(new Entry("a", 5000));
    out.add(new Entry("a", 3));
    out.add(new Entry("capa", 1));
    out.add(new Entry("capa", 1));
    return out;
  }

  static String results(List<LookupResult> res) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < res.size(); i++) {
      if (i > 0) b.append('\u001f');
      b.append(AnalysisRows.esc(res.get(i).key.toString())).append('\u001e').append(res.get(i).value);
    }
    return b.toString();
  }

  /** Every prefix of one to three code points of each term. */
  static Set<String> prefixes(List<Entry> entries) {
    Set<String> out = new LinkedHashSet<>();
    for (Entry e : entries) {
      String t = e.term();
      int end = 0;
      for (int k = 1; k <= 3 && end < t.length(); k++) {
        end += Character.charCount(t.codePointAt(end));
        out.add(t.substring(0, end));
      }
    }
    out.add("zzz");
    out.add("capaX");
    out.add("");
    return out;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("suggest");
    Files.createDirectories(out);
    List<Entry> entries = corpus();

    StringBuilder terms = new StringBuilder();
    for (Entry e : entries) {
      terms.append(AnalysisRows.esc(e.term())).append('\t').append(e.weight()).append('\n');
    }
    Files.writeString(out.resolve("terms.tsv"), terms.toString(), StandardCharsets.UTF_8);

    TSTLookup tst = new TSTLookup(new org.apache.lucene.store.ByteBuffersDirectory(), "tst");
    tst.build(new Entries(entries));

    StringBuilder rows = new StringBuilder();
    int[] nums = {1, 5, 50};
    for (String p : prefixes(entries)) {
      for (boolean more : new boolean[] {false, true}) {
        for (int num : nums) {
          rows.append(AnalysisRows.esc(p)).append('\t').append(more ? 1 : 0).append('\t').append(num).append('\t');
          String body;
          try {
            body = "R\t" + results(tst.lookup(p, more, num));
          } catch (RuntimeException e) {
            body = "X\t" + e.getClass().getSimpleName();
          }
          rows.append(body);
          rows.append('\n');
        }
      }
    }
    Files.writeString(out.resolve("tst.tsv"), rows.toString(), StandardCharsets.UTF_8);

    StringBuilder gets = new StringBuilder();
    Set<String> keys = new LinkedHashSet<>();
    for (Entry e : entries) keys.add(e.term());
    keys.add("missing");
    keys.add("ca");
    for (String k : keys) {
      if (k.isEmpty()) continue; // charAt(0) on an empty key throws
      Object v = tst.get(k);
      gets.append(AnalysisRows.esc(k)).append('\t').append(v == null ? "-" : v.toString()).append('\n');
    }
    Files.writeString(out.resolve("tst_get.tsv"), gets.toString(), StandardCharsets.UTF_8);

    try (OutputStream os = Files.newOutputStream(out.resolve("tst.bin"))) {
      tst.store(new OutputStreamDataOutput(os));
    }
    // load(store(x)) answers like x
    TSTLookup again = new TSTLookup(new org.apache.lucene.store.ByteBuffersDirectory(), "tst");
    try (var in = Files.newInputStream(out.resolve("tst.bin"))) {
      again.load(in);
    }
    if (again.getCount() != tst.getCount()) throw new IllegalStateException("count after load");
    // What Java's own load() gives back: a supplementary key is lost in store (each half is written
    // as U+FFFD), so this differs from tst_get.tsv for exactly those keys.
    StringBuilder reload = new StringBuilder();
    for (String k : keys) {
      if (k.isEmpty()) continue;
      Object v = again.get(k);
      reload.append(AnalysisRows.esc(k)).append('\t').append(v == null ? "-" : v.toString()).append('\n');
    }
    Files.writeString(out.resolve("tst_reload.tsv"), reload.toString(), StandardCharsets.UTF_8);
    System.out.println("GenSuggest: " + entries.size() + " entries, count " + tst.getCount());
  }
}
