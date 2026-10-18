import java.io.ByteArrayInputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.spell.Dictionary;
import org.apache.lucene.search.suggest.DocumentDictionary;
import org.apache.lucene.search.suggest.DocumentValueSourceDictionary;
import org.apache.lucene.search.suggest.FileDictionary;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.search.suggest.Lookup;
import org.apache.lucene.search.suggest.Lookup.LookupResult;
import org.apache.lucene.search.suggest.fst.FSTCompletionLookup;
import org.apache.lucene.search.suggest.fst.WFSTCompletionLookup;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.OutputStreamDataOutput;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.3a/b: {@code WFSTCompletionLookup}, {@code FSTCompletionLookup}, {@code FileDictionary},
 * {@code DocumentDictionary} and {@code DocumentValueSourceDictionary}, one fixture directory
 * {@code suggest_fst/}. The weighted-term corpus is {@code GenSuggest}'s ({@code terms.tsv}).
 * {@code wfst_*.tsv}/{@code fstc_*.tsv} are each lookup's answers to every short prefix; {@code
 * *.bin} are {@code store()}'s bytes; {@code dict_*.txt} are FileDictionary inputs with {@code
 * dict_*.tsv} their entries; {@code index/} is an index of {@code docs.tsv}'s documents with {@code
 * docdict_*.tsv} the entries of each DocumentDictionary over it.
 */
public class GenSuggestFst {
  static String results(List<LookupResult> res) {
    return GenSuggest.results(res);
  }

  static String run(java.util.concurrent.Callable<List<LookupResult>> c) {
    try {
      return "R\t" + results(c.call());
    } catch (Exception e) {
      return "X\t" + e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage()));
    }
  }

  static String hex(BytesRef b) {
    if (b == null) return "-";
    StringBuilder s = new StringBuilder();
    for (int i = 0; i < b.length; i++) s.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    return s.length() == 0 ? "." : s.toString();
  }

  static void lookups(
      Path out, String name, Lookup lk, List<GenSuggest.Entry> entries, boolean wfst)
      throws Exception {
    StringBuilder rows = new StringBuilder();
    int[] nums = {1, 3, 50};
    for (String p : GenSuggest.prefixes(entries)) {
      for (boolean flag : new boolean[] {false, true}) {
        // WFST only accepts onlyMorePopular=false; for FST the flag is higherWeightsFirst.
        if (wfst && flag) continue;
        for (int num : nums) {
          rows.append(AnalysisRows.esc(p)).append('\t').append(flag ? 1 : 0).append('\t').append(num).append('\t');
          rows.append(run(() -> lk.lookup(p, flag, num)));
          rows.append('\n');
        }
      }
    }
    Files.writeString(out.resolve(name + ".tsv"), rows.toString(), StandardCharsets.UTF_8);
  }

  static void gets(Path out, String name, java.util.function.Function<String, Object> get, List<GenSuggest.Entry> entries)
      throws Exception {
    StringBuilder b = new StringBuilder();
    Set<String> keys = new LinkedHashSet<>();
    for (GenSuggest.Entry e : entries) keys.add(e.term());
    keys.add("missing");
    keys.add("ca");
    keys.add("capaX");
    for (String k : keys) {
      Object v = get.apply(k);
      b.append(AnalysisRows.esc(k)).append('\t').append(v == null ? "-" : v.toString()).append('\n');
    }
    Files.writeString(out.resolve(name + ".tsv"), b.toString(), StandardCharsets.UTF_8);
  }

  static void store(Path out, String name, Lookup lk) throws Exception {
    try (OutputStream os = Files.newOutputStream(out.resolve(name + ".bin"))) {
      boolean ok = lk.store(new OutputStreamDataOutput(os));
      Files.writeString(out.resolve(name + ".stored"), ok ? "true\n" : "false\n");
    }
  }

  static String dictRows(Dictionary d) {
    StringBuilder b = new StringBuilder();
    try {
      InputIterator it = d.getEntryIterator();
      b.append("P\t").append(it.hasPayloads() ? 1 : 0).append('\t').append(it.hasContexts() ? 1 : 0).append('\n');
      BytesRef t;
      while ((t = it.next()) != null) {
        b.append("T\t").append(AnalysisRows.esc(t.utf8ToString())).append('\t').append(it.weight()).append('\t');
        b.append(it.hasPayloads() ? hex(it.payload()) : "-").append('\t');
        if (it.hasContexts()) {
          TreeSet<String> cs = new TreeSet<>();
          for (BytesRef c : it.contexts()) cs.add(hex(c));
          b.append(cs.isEmpty() ? "." : String.join(",", cs));
        } else {
          b.append('-');
        }
        b.append('\n');
      }
    } catch (Exception e) {
      return "X\t" + e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(String.valueOf(e.getMessage())) + "\n";
    }
    return b.toString();
  }

  static void fileDictionaries(Path out) throws Exception {
    String[][] inputs = {
      {"dict_payload", "apple\t10\tred\nbanana\t7\tyellow\ncherry\t3\nfig\nlong\t12345678901\tbig\nfloaty\t12.9\tf\ngrape\t1e3\tg\nnegfloat\t-2.5\tn\n日本\t5\t語\n\t7\tleading\ntrail\t4\t\n"},
      {"dict_plain", "alpha\t3\nbeta\t4\tignored payload\ngamma\n\ndelta\t-1\r\nepsilon\t9\rzeta\t2"},
      {"dict_spaces", "one\t  5  \ntwo\t7d\nthree\t8.5f\nfour\tNaN\nfive\tInfinity\nsix\t-Infinity\n"},
      {"dict_empty", ""},
      {"dict_only_term", "solo"},
      {"dict_bad4", "a\t1\tp\textra\n"},
      {"dict_badweight", "a\t1\nb\tnotanumber\n"},
      {"dict_badfirst", "a\tx\n"},
      {"dict_alltabs", "\t\t\n"},
      // Double.parseDouble spells NaN and Infinity in one case only; it reports
      // the trimmed input, and "empty String" for an empty weight
      {"dict_lowernan", "a\t1\nb\tnan\n"},
      {"dict_upperinf", "a\t1\nb\tINFINITY\n"},
      {"dict_padbad", "a\t1\nb\t x \n"},
      {"dict_emptyweight", "a\t1\tp\nb\t\tq\n"},
    };
    for (String[] in : inputs) {
      Files.write(out.resolve(in[0] + ".txt"), in[1].getBytes(StandardCharsets.UTF_8));
      FileDictionary d = new FileDictionary(new ByteArrayInputStream(in[1].getBytes(StandardCharsets.UTF_8)));
      Files.writeString(out.resolve(in[0] + ".tsv"), dictRows(d), StandardCharsets.UTF_8);
    }
    // a delimiter other than the tab
    String semi = "x;5;px\ny;6;py\nz\n";
    Files.write(out.resolve("dict_semicolon.txt"), semi.getBytes(StandardCharsets.UTF_8));
    FileDictionary d = new FileDictionary(new ByteArrayInputStream(semi.getBytes(StandardCharsets.UTF_8)), ";");
    Files.writeString(out.resolve("dict_semicolon.tsv"), dictRows(d), StandardCharsets.UTF_8);
  }

  static void documentDictionaries(Path out) throws Exception {
    Path index = out.resolve("index");
    Files.createDirectories(index);
    try (var old = Files.list(index)) {
      for (Path p : old.toList()) Files.delete(p);
    }
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    StringBuilder docs = new StringBuilder();
    try (Directory dir = FSDirectory.open(index);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < 40; i++) {
        Document d = new Document();
        d.add(new StringField("id", Integer.toString(i), Field.Store.YES));
        // a few docs without a title; some with two
        if (i % 9 != 4) d.add(new StoredField("title", "title " + (i * 7 % 13)));
        if (i % 6 == 1) d.add(new StoredField("title", "also " + i));
        if (i == 11) {
          // a numeric title: its stringValue() is the number's text
          Document n = new Document();
          n.add(new StringField("id", "11", Field.Store.YES));
          n.add(new StoredField("title", 77L));
          n.add(new StoredField("title", "after the number"));
          n.add(new StoredField("weight", 999L));
          d = n;
        } else {
          if (i % 5 != 3) d.add(new StoredField("weight", (long) (i * 31 % 17)));
          d.add(new NumericDocValuesField("w2", (long) (i * 13 % 23)));
        }
        // numbers as titles: stringValue() is Double.toString / Float.toString
        switch (i) {
          case 20 -> d.add(new StoredField("title", 12345678.0));
          case 21 -> d.add(new StoredField("title", 0.001));
          case 22 -> d.add(new StoredField("title", 9999999.0));
          case 23 -> d.add(new StoredField("title", 1e-4));
          case 24 -> d.add(new StoredField("title", -2.5f));
          case 25 -> d.add(new StoredField("title", 100.0f));
          case 26 -> d.add(new StoredField("title", 1.0f / 3));
          case 27 -> d.add(new StoredField("title", 0.1 + 0.2));
          case 28 -> d.add(new StoredField("title", 1.0E22));
          case 29 -> d.add(new StoredField("title", 7));
          case 30 -> d.add(new StoredField("title", 0.0));
          case 31 -> d.add(new StoredField("title", -0.0f));
          case 32 -> d.add(new StoredField("title", Double.NaN));
          case 33 -> d.add(new StoredField("title", Float.NEGATIVE_INFINITY));
          default -> {}
        }
        if (i % 4 == 0) d.add(new StoredField("payload", new BytesRef(new byte[] {(byte) i, 0, (byte) 0xff})));
        else if (i % 4 == 1) d.add(new StoredField("payload", "pay " + i));
        else if (i % 4 == 2) d.add(new StoredField("payload", (long) i));
        if (i % 3 != 0) d.add(new StoredField("ctx", "c" + (i % 3)));
        if (i % 7 == 0) d.add(new StoredField("ctx", new BytesRef("bin" + i)));
        if (i % 7 == 1) d.add(new StoredField("ctx", (long) i));
        w.addDocument(d);
        docs.append(i).append('\n');
        if (i == 14 || i == 29) w.commit();
      }
      for (int i = 2; i < 40; i += 5) w.deleteDocuments(new Term("id", Integer.toString(i)));
      w.commit();
    }
    Files.writeString(out.resolve("docs.tsv"), docs.toString(), StandardCharsets.UTF_8);
    try (Directory dir = FSDirectory.open(index);
        DirectoryReader r = DirectoryReader.open(dir)) {
      Files.writeString(
          out.resolve("docdict_weight.tsv"),
          dictRows(new DocumentDictionary(r, "title", "weight")),
          StandardCharsets.UTF_8);
      Files.writeString(
          out.resolve("docdict_dv.tsv"),
          dictRows(new DocumentDictionary(r, "title", "w2")),
          StandardCharsets.UTF_8);
      Files.writeString(
          out.resolve("docdict_full.tsv"),
          dictRows(new DocumentDictionary(r, "title", "weight", "payload", "ctx")),
          StandardCharsets.UTF_8);
      Files.writeString(
          out.resolve("docdict_none.tsv"),
          dictRows(new DocumentDictionary(r, "title", "missing")),
          StandardCharsets.UTF_8);
      Files.writeString(
          out.resolve("docdict_nofield.tsv"),
          dictRows(new DocumentDictionary(r, "nofield", "weight")),
          StandardCharsets.UTF_8);
      Files.writeString(
          out.resolve("docvs_dv.tsv"),
          dictRows(new DocumentValueSourceDictionary(r, "title", LongValuesSource.fromLongField("w2"))),
          StandardCharsets.UTF_8);
      Files.writeString(
          out.resolve("docvs_full.tsv"),
          dictRows(
              new DocumentValueSourceDictionary(
                  r, "title", LongValuesSource.constant(5), "payload", "ctx")),
          StandardCharsets.UTF_8);
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("suggest_fst");
    Files.createDirectories(out);
    List<GenSuggest.Entry> entries = GenSuggest.corpus();
    StringBuilder terms = new StringBuilder();
    for (GenSuggest.Entry e : entries) {
      terms.append(AnalysisRows.esc(e.term())).append('\t').append(e.weight()).append('\n');
    }
    Files.writeString(out.resolve("terms.tsv"), terms.toString(), StandardCharsets.UTF_8);

    // --- WFSTCompletionLookup
    for (boolean exactFirst : new boolean[] {true, false}) {
      String tag = exactFirst ? "exact" : "plain";
      WFSTCompletionLookup w = new WFSTCompletionLookup(new ByteBuffersDirectory(), "wfst", exactFirst);
      w.build(new GenSuggest.Entries(entries));
      lookups(out, "wfst_" + tag, w, entries, true);
      gets(out, "wfst_get_" + tag, w::get, entries);
      store(out, "wfst_" + tag, w);
      Files.writeString(out.resolve("wfst_" + tag + ".count"), w.getCount() + "\n");
      WFSTCompletionLookup again = new WFSTCompletionLookup(new ByteBuffersDirectory(), "wfst", exactFirst);
      try (var in = Files.newInputStream(out.resolve("wfst_" + tag + ".bin"))) {
        again.load(in);
      }
      lookups(out, "wfst_" + tag + "_reload", again, entries, true);
    }
    // an empty build stores a count and refuses to save
    WFSTCompletionLookup empty = new WFSTCompletionLookup(new ByteBuffersDirectory(), "wfst");
    empty.build(new GenSuggest.Entries(new ArrayList<>()));
    store(out, "wfst_empty", empty);
    Files.writeString(out.resolve("wfst_empty.lookup"), run(() -> empty.lookup("a", false, 3)) + "\n");
    // error cases
    StringBuilder errs = new StringBuilder();
    WFSTCompletionLookup ew = new WFSTCompletionLookup(new ByteBuffersDirectory(), "wfst");
    ew.build(new GenSuggest.Entries(entries));
    errs.append("only_more_popular\t").append(run(() -> ew.lookup("ca", true, 3))).append('\n');
    errs.append("contexts\t").append(run(() -> ew.lookup("ca", new java.util.HashSet<>(List.of(new BytesRef("c"))), false, 3))).append('\n');
    for (long weight : new long[] {-1L, (long) Integer.MAX_VALUE + 1}) {
      List<GenSuggest.Entry> bad = List.of(new GenSuggest.Entry("a", weight));
      String r;
      try {
        new WFSTCompletionLookup(new ByteBuffersDirectory(), "wfst").build(new GenSuggest.Entries(bad));
        r = "built";
      } catch (Exception e) {
        r = e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(e.getMessage());
      }
      errs.append("weight_").append(weight).append('\t').append(r).append('\n');
    }
    Files.writeString(out.resolve("wfst_errors.tsv"), errs.toString(), StandardCharsets.UTF_8);
    // the heaviest duplicate wins, and a weight of zero and of Integer.MAX_VALUE round trip
    List<GenSuggest.Entry> edge = List.of(
        new GenSuggest.Entry("dup", 5), new GenSuggest.Entry("dup", 9), new GenSuggest.Entry("dup", 1),
        new GenSuggest.Entry("zero", 0), new GenSuggest.Entry("max", Integer.MAX_VALUE),
        new GenSuggest.Entry("dupx", 9));
    WFSTCompletionLookup ed = new WFSTCompletionLookup(new ByteBuffersDirectory(), "wfst");
    ed.build(new GenSuggest.Entries(edge));
    StringBuilder edgeRows = new StringBuilder();
    for (String p : new String[] {"dup", "d", "zero", "max", "m", "z"}) {
      for (int num : new int[] {1, 2, 5}) {
        edgeRows.append(p).append('\t').append(num).append('\t').append(run(() -> ed.lookup(p, false, num))).append('\n');
      }
    }
    edgeRows.append("count\t").append(ed.getCount()).append('\n');
    Files.writeString(out.resolve("wfst_edge.tsv"), edgeRows.toString(), StandardCharsets.UTF_8);

    // --- FSTCompletionLookup: ten buckets exact first, and three buckets without
    int[] bucketCounts = {10, 3, 1};
    boolean[] exacts = {true, false, true};
    for (int k = 0; k < bucketCounts.length; k++) {
      String tag = "b" + bucketCounts[k] + (exacts[k] ? "e" : "n");
      FSTCompletionLookup f = new FSTCompletionLookup(new ByteBuffersDirectory(), "fstc", bucketCounts[k], exacts[k]);
      f.build(new GenSuggest.Entries(entries));
      lookups(out, "fstc_" + tag, f, entries, false);
      gets(out, "fstc_get_" + tag, f::get, entries);
      store(out, "fstc_" + tag, f);
      Files.writeString(out.resolve("fstc_" + tag + ".count"), f.getCount() + "\n");
      FSTCompletionLookup again = new FSTCompletionLookup(new ByteBuffersDirectory(), "fstc", bucketCounts[k], exacts[k]);
      try (var in = Files.newInputStream(out.resolve("fstc_" + tag + ".bin"))) {
        again.load(in);
      }
      lookups(out, "fstc_" + tag + "_reload", again, entries, false);
    }
    FSTCompletionLookup fe = new FSTCompletionLookup(new ByteBuffersDirectory(), "fstc");
    fe.build(new GenSuggest.Entries(new ArrayList<>()));
    store(out, "fstc_empty", fe);
    // mixed-sign weights: the weight prefix is a big-endian int compared as
    // unsigned bytes, so the negative weights land in the highest buckets
    List<GenSuggest.Entry> signed = List.of(
        new GenSuggest.Entry("neg", -5), new GenSuggest.Entry("negone", -1),
        new GenSuggest.Entry("nil", 0), new GenSuggest.Entry("nine", 9),
        new GenSuggest.Entry("ninety", 90), new GenSuggest.Entry("nmin", Integer.MIN_VALUE),
        new GenSuggest.Entry("nmax", Integer.MAX_VALUE), new GenSuggest.Entry("nope", 3));
    FSTCompletionLookup fs = new FSTCompletionLookup(new ByteBuffersDirectory(), "fstc", 4, true);
    fs.build(new GenSuggest.Entries(signed));
    lookups(out, "fstc_signed", fs, signed, false);
    gets(out, "fstc_get_signed", fs::get, signed);
    store(out, "fstc_signed", fs);
    StringBuilder ferrs = new StringBuilder();
    ferrs.append("contexts\t").append(run(() -> fe.lookup("a", new java.util.HashSet<>(List.of(new BytesRef("c"))), false, 3))).append('\n');
    for (long weight : new long[] {Long.MIN_VALUE, (long) Integer.MAX_VALUE + 1}) {
      List<GenSuggest.Entry> bad = List.of(new GenSuggest.Entry("a", weight));
      String r;
      try {
        new FSTCompletionLookup(new ByteBuffersDirectory(), "fstc").build(new GenSuggest.Entries(bad));
        r = "built";
      } catch (Exception e) {
        r = e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(e.getMessage());
      }
      ferrs.append("weight_").append(weight).append('\t').append(r).append('\n');
    }
    for (int b : new int[] {0, 256, -1}) {
      String r;
      try {
        new org.apache.lucene.search.suggest.fst.FSTCompletionBuilder(
            b, new org.apache.lucene.search.suggest.InMemorySorter(java.util.Comparator.naturalOrder()), 5);
        r = "built";
      } catch (Exception e) {
        r = e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(e.getMessage());
      }
      ferrs.append("buckets_").append(b).append('\t').append(r).append('\n');
    }
    {
      var builder = new org.apache.lucene.search.suggest.fst.FSTCompletionBuilder(4,
          new org.apache.lucene.search.suggest.InMemorySorter(java.util.Comparator.naturalOrder()), 5);
      for (int b : new int[] {-1, 4}) {
        String r;
        try {
          builder.add(new BytesRef("x"), b);
          r = "added";
        } catch (Exception e) {
          r = e.getClass().getSimpleName() + "\t" + AnalysisRows.esc(e.getMessage());
        }
        ferrs.append("add_bucket_").append(b).append('\t').append(r).append('\n');
      }
    }
    Files.writeString(out.resolve("fstc_errors.tsv"), ferrs.toString(), StandardCharsets.UTF_8);

    // java.util.Random, which UnsortedInputIterator shuffles with
    StringBuilder rnd = new StringBuilder();
    for (long seed : new long[] {0L, 42L, -7L, 1305L, Long.MAX_VALUE}) {
      for (int bound : new int[] {1, 2, 3, 10, 16, 100, 1000, 1 << 20, 1_000_000_007, Integer.MAX_VALUE}) {
        java.util.Random random = new java.util.Random(seed);
        rnd.append(seed).append('\t').append(bound);
        for (int i = 0; i < 6; i++) rnd.append('\t').append(random.nextInt(bound));
        rnd.append('\n');
      }
    }
    Files.writeString(out.resolve("random.tsv"), rnd.toString());

    fileDictionaries(out);
    documentDictionaries(out);
    System.out.println("GenSuggestFst: " + entries.size() + " entries");
  }
}
