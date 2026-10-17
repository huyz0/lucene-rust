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
import org.apache.lucene.search.suggest.analyzing.AnalyzingSuggester;
import org.apache.lucene.search.suggest.analyzing.FuzzySuggester;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.3c's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_analyzing.rs}):
 * {@code GenAnalyzingSuggest}'s 32 phrases ({@code fixtures/data/suggest_analyzing/phrases.tsv}).
 * {@code analyzing_lookup}/{@code fuzzy_lookup}: top 5 for every one to three word prefix (and
 * each phrase cut after four characters, so the fuzzy suggester has typos to forgive), units: lookups;
 * {@code analyzing_build}/{@code fuzzy_build}: a build over all phrases, units: phrases. Output is
 * {@code name\tns_per_unit\tunits}.
 */
public class AnalyzingSuggestMicro {
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

  static AnalyzingSuggester analyzing() {
    return new AnalyzingSuggester(new ByteBuffersDirectory(), "an", new StandardAnalyzer(stops()));
  }

  static FuzzySuggester fuzzy() {
    return new FuzzySuggester(new ByteBuffersDirectory(), "fz", new StandardAnalyzer(stops()));
  }

  public static void main(String[] args) throws Exception {
    List<Entry> entries = new ArrayList<>();
    for (String line :
        Files.readAllLines(
            Path.of("fixtures/data/suggest_analyzing/phrases.tsv"), StandardCharsets.UTF_8)) {
      String[] c = line.split("\t");
      entries.add(new Entry(c[0], Long.parseLong(c[1])));
    }
    List<String> queries = queries(entries);
    AnalyzingSuggester a = analyzing();
    a.build(new Entries(entries));
    FuzzySuggester f = fuzzy();
    f.build(new Entries(entries));
    lookups("analyzing_lookup", a, queries);
    lookups("fuzzy_lookup", f, queries);
    SweepMicro.measure("analyzing_build", () -> {
      try {
        AnalyzingSuggester b = analyzing();
        b.build(new Entries(entries));
        SweepMicro.sink += b.getCount();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return entries.size();
    });
    SweepMicro.measure("fuzzy_build", () -> {
      try {
        FuzzySuggester b = fuzzy();
        b.build(new Entries(entries));
        SweepMicro.sink += b.getCount();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return entries.size();
    });
  }
}
