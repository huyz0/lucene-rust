import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.spell.DirectSpellChecker;
import org.apache.lucene.search.spell.HighFrequencyDictionary;
import org.apache.lucene.search.spell.JaroWinklerDistance;
import org.apache.lucene.search.spell.LevenshteinDistance;
import org.apache.lucene.search.spell.LuceneDictionary;
import org.apache.lucene.search.spell.LuceneLevenshteinDistance;
import org.apache.lucene.search.spell.NGramDistance;
import org.apache.lucene.search.spell.SuggestMode;
import org.apache.lucene.search.spell.SuggestWord;
import org.apache.lucene.search.spell.SuggestWordFrequencyComparator;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.3e: {@code DirectSpellChecker}, {@code LuceneDictionary} and {@code
 * HighFrequencyDictionary} over a four-segment index with deletions. Writes {@code
 * spell_checker/}: {@code index/}, {@code queries.tsv}, {@code direct.tsv} (configuration, query,
 * mode, numSug, then {@code string:freq:scoreBits} per suggestion), {@code dictionaries.tsv}
 * (dictionary, then {@code hexTerm:weight} per entry) and {@code errors.tsv} (setter, value,
 * exception class: message).
 */
public class GenSpellChecker {
  static final Object[][] VOCABULARY = {
    {"lucene", 20}, {"lucine", 3}, {"lucen", 2}, {"lucenes", 1}, {"licene", 1}, {"lucent", 4},
    {"lugene", 1}, {"search", 15}, {"serch", 1}, {"seach", 2}, {"searches", 5}, {"searcher", 3},
    {"apache", 12}, {"apachee", 1}, {"apeche", 1}, {"index", 18}, {"indexs", 1}, {"indexes", 6},
    {"indez", 1}, {"spell", 9}, {"spel", 2}, {"spelling", 4}, {"smell", 3}, {"shell", 5},
    {"spill", 1}, {"checker", 7}, {"chekcer", 1}, {"checkers", 2}, {"café", 3}, {"cafe", 4},
    {"caffe", 1}, {"naïve", 2}, {"naive", 3}, {"日本語", 2}, {"日本", 3}, {"a", 30}, {"ab", 2},
    {"abc", 1}, {"word", 6}, {"wrod", 1}, {"world", 8}, {"sword", 2}, {"words", 3},
    {"breakfast", 2}, {"break", 5}, {"fast", 7}, {"breakfasts", 1}, {"i̇ndex", 2}, {"😀xy", 2},
    {"😀xz", 3},
  };

  static final String[] QUERIES = {
    "lucene", "lucenee", "lcuene", "LUCENE", "Lucene", "lucen", "luc", "lu", "searh", "serach",
    "apahce", "indx", "indexx", "spel", "spelll", "chekcer", "cafe", "cafè", "caffé", "naive",
    "naïv", "日本人", "wrold", "wordd", "brekfast", "abc", "abcd", "xyzzy", "search", "breakfast",
    "ndex", "İNDEX", "😀xx", "a", "smel", "aplle", "lucenes",
  };

  static final int N = 60;

  record Config(String name, Consumer<DirectSpellChecker> setup, Float accuracyArg) {}

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("spell_checker");
    Path indexDir = out.resolve("index");
    if (Files.exists(indexDir)) IOUtils.rm(indexDir);
    Files.createDirectories(indexDir);

    List<StringBuilder> docs = new ArrayList<>();
    for (int d = 0; d < N; d++) docs.add(new StringBuilder());
    for (int i = 0; i < VOCABULARY.length; i++) {
      String w = (String) VOCABULARY[i][0];
      int c = (Integer) VOCABULARY[i][1];
      for (int k = 0; k < c; k++) docs.get((i * 13 + k * 7) % N).append(w).append(' ');
    }
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new WhitespaceAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int d = 0; d < N; d++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(d), Field.Store.NO));
          doc.add(new TextField("body", docs.get(d).toString(), Field.Store.NO));
          if (d % 5 == 0) doc.add(new TextField("other", "lucene other", Field.Store.NO));
          w.addDocument(doc);
          if (d % 15 == 14) w.commit();
        }
        // deleted documents still count in docFreq, not in numDocs
        for (int d = 15; d < 30; d += 3) w.deleteDocuments(new Term("id", Integer.toString(d)));
        w.commit();
      }
    }

    List<Config> configs = new ArrayList<>();
    configs.add(new Config("default", s -> {}, null));
    configs.add(new Config("max_edits_1", s -> s.setMaxEdits(1), null));
    configs.add(new Config("min_prefix_0", s -> s.setMinPrefix(0), null));
    configs.add(new Config("min_prefix_2", s -> s.setMinPrefix(2), null));
    configs.add(new Config("inspections_1", s -> s.setMaxInspections(1), null));
    configs.add(new Config("inspections_20", s -> s.setMaxInspections(20), null));
    configs.add(new Config("accuracy_0", s -> s.setAccuracy(0f), null));
    configs.add(new Config("accuracy_0.8", s -> s.setAccuracy(0.8f), null));
    configs.add(new Config("accuracy_arg_0.3", s -> {}, 0.3f));
    configs.add(new Config("threshold_0.05", s -> s.setThresholdFrequency(0.05f), null));
    configs.add(new Config("threshold_3", s -> s.setThresholdFrequency(3f), null));
    configs.add(new Config("max_query_freq_0", s -> s.setMaxQueryFrequency(0f), null));
    configs.add(new Config("max_query_freq_0.5", s -> s.setMaxQueryFrequency(0.5f), null));
    configs.add(new Config("max_query_freq_5", s -> s.setMaxQueryFrequency(5f), null));
    configs.add(
        new Config(
            "query_length_1_5",
            s -> {
              s.setMinQueryLength(1);
              s.setMaxQueryLength(5);
            },
            null));
    configs.add(new Config("no_lowercase", s -> s.setLowerCaseTerms(false), null));
    configs.add(
        new Config("frequency_comparator", s -> s.setComparator(new SuggestWordFrequencyComparator()), null));
    configs.add(new Config("distance_levenshtein", s -> s.setDistance(new LevenshteinDistance()), null));
    configs.add(new Config("distance_jarowinkler", s -> s.setDistance(new JaroWinklerDistance()), null));
    configs.add(new Config("distance_ngram", s -> s.setDistance(new NGramDistance()), null));
    configs.add(
        new Config("distance_lucene_levenshtein", s -> s.setDistance(new LuceneLevenshteinDistance()), null));
    configs.add(
        new Config(
            "mixed",
            s -> {
              s.setMaxEdits(1);
              s.setMinPrefix(0);
              s.setAccuracy(0.3f);
              s.setMaxInspections(2);
              s.setDistance(new NGramDistance(3));
              s.setComparator(new SuggestWordFrequencyComparator());
              s.setThresholdFrequency(2f);
              s.setMaxQueryFrequency(0.9f);
            },
            null));

    StringBuilder qs = new StringBuilder();
    for (String q : QUERIES) qs.append(q).append('\n');
    Files.writeString(out.resolve("queries.tsv"), qs.toString(), StandardCharsets.UTF_8);

    StringBuilder rows = new StringBuilder();
    StringBuilder dicts = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        IndexReader reader = DirectoryReader.open(dir)) {
      for (Config c : configs) {
        DirectSpellChecker s = new DirectSpellChecker();
        c.setup.accept(s);
        for (String q : QUERIES) {
          for (SuggestMode mode : SuggestMode.values()) {
            for (int num : new int[] {1, 3, 10}) {
              Term t = new Term("body", q);
              SuggestWord[] got =
                  c.accuracyArg == null
                      ? s.suggestSimilar(t, num, reader, mode)
                      : s.suggestSimilar(t, num, reader, mode, c.accuracyArg);
              rows.append(c.name).append('\t').append(q).append('\t').append(mode).append('\t').append(num);
              for (SuggestWord w : got) {
                rows.append('\t')
                    .append(w.string)
                    .append(':')
                    .append(w.freq)
                    .append(':')
                    .append(Integer.toHexString(Float.floatToIntBits(w.score)));
              }
              rows.append('\n');
            }
          }
        }
      }
      // a field no document has, and a field only some have
      for (String field : new String[] {"missing", "other"}) {
        DirectSpellChecker s = new DirectSpellChecker();
        SuggestWord[] got = s.suggestSimilar(new Term(field, "lucenee"), 5, reader);
        rows.append("field_").append(field).append("\tlucenee\tSUGGEST_WHEN_NOT_IN_INDEX\t5");
        for (SuggestWord w : got) rows.append('\t').append(w.string).append(':').append(w.freq);
        rows.append('\n');
      }

      dump(dicts, "lucene_body", new LuceneDictionary(reader, "body").getEntryIterator());
      dump(dicts, "lucene_other", new LuceneDictionary(reader, "other").getEntryIterator());
      dump(dicts, "lucene_missing", new LuceneDictionary(reader, "missing").getEntryIterator());
      for (float th : new float[] {0f, 0.05f, 0.1f, 0.3f, 2f}) {
        dump(
            dicts,
            "high_frequency_body_" + th,
            new HighFrequencyDictionary(reader, "body", th).getEntryIterator());
      }
      dump(
          dicts,
          "high_frequency_missing",
          new HighFrequencyDictionary(reader, "missing", 0f).getEntryIterator());
    }
    Files.writeString(out.resolve("direct.tsv"), rows.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("dictionaries.tsv"), dicts.toString(), StandardCharsets.UTF_8);

    StringBuilder errs = new StringBuilder();
    error(errs, "max_edits", "0", s -> s.setMaxEdits(0));
    error(errs, "max_edits", "3", s -> s.setMaxEdits(3));
    error(errs, "max_edits", "2", s -> s.setMaxEdits(2));
    error(errs, "threshold_frequency", "1.5", s -> s.setThresholdFrequency(1.5f));
    error(errs, "threshold_frequency", "2", s -> s.setThresholdFrequency(2f));
    error(errs, "threshold_frequency", "0.5", s -> s.setThresholdFrequency(0.5f));
    error(errs, "max_query_frequency", "2.5", s -> s.setMaxQueryFrequency(2.5f));
    error(errs, "max_query_frequency", "3", s -> s.setMaxQueryFrequency(3f));
    error(errs, "min_query_length", "10_after_max_5", s -> {
      s.setMaxQueryLength(5);
      s.setMinQueryLength(10);
    });
    error(errs, "max_query_length", "2", s -> s.setMaxQueryLength(2));
    error(errs, "max_query_length", "4", s -> s.setMaxQueryLength(4));
    Files.writeString(out.resolve("errors.tsv"), errs.toString(), StandardCharsets.UTF_8);
  }

  static void error(StringBuilder errs, String setter, String value, Consumer<DirectSpellChecker> f) {
    DirectSpellChecker s = new DirectSpellChecker();
    errs.append(setter).append('\t').append(value).append('\t');
    try {
      f.accept(s);
      errs.append("ok");
    } catch (RuntimeException e) {
      errs.append(e.getClass().getSimpleName()).append(": ").append(e.getMessage());
    }
    errs.append('\n');
  }

  static void dump(StringBuilder sb, String name, InputIterator it) throws Exception {
    sb.append(name).append('\t').append(it.hasPayloads()).append('\t').append(it.hasContexts());
    BytesRef b;
    while ((b = it.next()) != null) {
      sb.append('\t');
      for (int i = 0; i < b.length; i++) sb.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
      sb.append(':').append(it.weight());
    }
    sb.append('\n');
  }
}
