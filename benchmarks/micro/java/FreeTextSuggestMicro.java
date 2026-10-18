import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.search.suggest.Lookup;
import org.apache.lucene.search.suggest.analyzing.FreeTextSuggester;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.3c's FreeText pair (the Rust twin is {@code benchmarks/rust-runner/src/micro_freetext.rs}) over the phrases and
 * queries of {@code AnalyzingSuggestMicro}. {@code freetext_lookup}: top 5 for every query with a token, units: lookups;
 * {@code freetext_build}: a build over all phrases, units: phrases. Output is {@code name	ns_per_unit	units}.
 */
public class FreeTextSuggestMicro {
  record Entry(String term, long weight) {}

  static final class Entries implements InputIterator {
    final List<Entry> entries;
    int i = -1;

    Entries(List<Entry> entries) {
      this.entries = entries;
    }

    @Override
    public BytesRef next() {
      return ++i < entries.size() ? new BytesRef(entries.get(i).term()) : null;
    }

    @Override
    public long weight() {
      return entries.get(i).weight();
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

  /** Word prefixes of each phrase, and each phrase cut after four characters. */
  static List<String> queries(List<Entry> entries) {
    Set<String> out = new LinkedHashSet<>();
    for (Entry e : entries) {
      String lower = e.term().toLowerCase(java.util.Locale.ROOT);
      String[] words = lower.split(" ");
      StringBuilder acc = new StringBuilder();
      for (int i = 0; i < words.length && i < 3; i++) {
        if (i > 0) acc.append(' ');
        acc.append(words[i]);
        out.add(acc.toString());
      }
      int cut = Math.min(4, lower.codePointCount(0, lower.length()));
      out.add(lower.substring(0, lower.offsetByCodePoints(0, cut)));
    }
    return new ArrayList<>(out);
  }

  static CharArraySet stops() {
    return new CharArraySet(List.of("the", "of", "a"), false);
  }

  static void lookups(String name, Lookup lk, List<String> queries) throws Exception {
    SweepMicro.measure(name, () -> {
      for (String q : queries) {
        try {
          SweepMicro.sink += lk.lookup(q, false, 5).size();
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
      return queries.size();
    });
  }

  static FreeTextSuggester make() {
    return new FreeTextSuggester(new StandardAnalyzer(stops()));
  }

  public static void main(String[] args) throws Exception {
    List<Entry> entries = new ArrayList<>();
    for (String line :
        Files.readAllLines(
            Path.of("fixtures/data/suggest_analyzing/phrases.tsv"), StandardCharsets.UTF_8)) {
      String[] c = line.split("	");
      entries.add(new Entry(c[0], Long.parseLong(c[1])));
    }
    List<String> queries = queries(entries);
    FreeTextSuggester s = make();
    s.build(new Entries(entries));
    // a query of stop words alone analyzes to no tokens, which both engines
    // refuse: time the others
    queries.removeIf(
        q -> {
          try {
            s.lookup(q, false, 5);
            return false;
          } catch (Exception e) {
            return true;
          }
        });
    lookups("freetext_lookup", s, queries);
    SweepMicro.measure("freetext_build", () -> {
      try {
        FreeTextSuggester b = make();
        b.build(new Entries(entries));
        SweepMicro.sink += b.getCount();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return entries.size();
    });
  }
}
