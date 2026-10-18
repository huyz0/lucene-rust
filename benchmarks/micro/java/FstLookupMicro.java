import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.search.suggest.Lookup;
import org.apache.lucene.search.suggest.fst.FSTCompletionLookup;
import org.apache.lucene.search.suggest.fst.WFSTCompletionLookup;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.3b's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_fst_lookup.rs}):
 * {@code GenSuggestFst}'s 324 weighted terms ({@code fixtures/data/suggest_fst/terms.tsv}). {@code
 * wfst_lookup}/{@code fstc_lookup}: top 10 for every 1-3 character prefix of a term (WFST exact
 * first; FSTCompletion with ten buckets, higher weights first), units: lookups; {@code
 * wfst_build}/{@code fstc_build}: a build over all terms, units: terms. Output is {@code
 * name\tns_per_unit\tunits}.
 */
public class FstLookupMicro {
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

  static List<String> prefixes(List<Entry> entries) {
    Set<String> out = new LinkedHashSet<>();
    for (Entry e : entries) {
      String t = e.term();
      int end = 0;
      for (int k = 1; k <= 3 && end < t.length(); k++) {
        end += Character.charCount(t.codePointAt(end));
        out.add(t.substring(0, end));
      }
    }
    return new ArrayList<>(out);
  }

  static void lookups(String name, Lookup lk, List<String> prefixes, boolean flag)
      throws Exception {
    SweepMicro.measure(name, () -> {
      for (String p : prefixes) {
        try {
          SweepMicro.sink += lk.lookup(p, flag, 10).size();
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
      return prefixes.size();
    });
  }

  public static void main(String[] args) throws Exception {
    List<Entry> entries = new ArrayList<>();
    for (String line :
        Files.readAllLines(Path.of("fixtures/data/suggest_fst/terms.tsv"), StandardCharsets.UTF_8)) {
      String[] c = line.split("\t");
      entries.add(new Entry(c[0], Long.parseLong(c[1])));
    }
    List<String> prefixes = prefixes(entries);
    WFSTCompletionLookup w = new WFSTCompletionLookup(new ByteBuffersDirectory(), "wfst", true);
    w.build(new Entries(entries));
    FSTCompletionLookup f = new FSTCompletionLookup(new ByteBuffersDirectory(), "fstc", 10, true);
    f.build(new Entries(entries));
    lookups("wfst_lookup", w, prefixes, false);
    lookups("fstc_lookup", f, prefixes, true);
    SweepMicro.measure("wfst_build", () -> {
      try {
        WFSTCompletionLookup b = new WFSTCompletionLookup(new ByteBuffersDirectory(), "wfst", true);
        b.build(new Entries(entries));
        SweepMicro.sink += b.getCount();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return entries.size();
    });
    SweepMicro.measure("fstc_build", () -> {
      try {
        FSTCompletionLookup b =
            new FSTCompletionLookup(new ByteBuffersDirectory(), "fstc", 10, true);
        b.build(new Entries(entries));
        SweepMicro.sink += b.getCount();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return entries.size();
    });
  }
}
