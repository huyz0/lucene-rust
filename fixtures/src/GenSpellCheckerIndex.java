import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.spell.JaroWinklerDistance;
import org.apache.lucene.search.spell.LuceneDictionary;
import org.apache.lucene.search.spell.LuceneLevenshteinDistance;
import org.apache.lucene.search.spell.NGramDistance;
import org.apache.lucene.search.spell.PlainTextDictionary;
import org.apache.lucene.search.spell.SpellChecker;
import org.apache.lucene.search.spell.SuggestMode;
import org.apache.lucene.search.spell.SuggestWordFrequencyComparator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.3e: the classic {@code SpellChecker}. Writes {@code spell_checker_index/}: {@code
 * user/} (a one-segment user index, field {@code body}), {@code first/} (a spell index built from
 * {@code DICTIONARY} by a {@code PlainTextDictionary}, fully merged), {@code second/} (the same,
 * then the user index's {@code LuceneDictionary} added unmerged), {@code suggest.tsv} (stage,
 * configuration, query, numSug, mode or {@code -} without the user index, then one suggestion per
 * column) and {@code exist.tsv} (stage, word, exists).
 */
public class GenSpellCheckerIndex {
  static final String DICTIONARY =
      "lucene\nlucine\nlucent\nsearch\nsearches\nsearcher\nseach\napache\napeche\nindex\nindexes\n"
          + "spell\nspelling\nsmell\nshell\nspill\nchecker\ncheckers\ncafé\ncafe\ncaffe\nnaïve\n"
          + "naive\n日本語\nab\nabc\nabcd\nword\nworld\nsword\nwords\nbreakfast\nbreak\nfast\n"
          + "lucene\n😀xyz\n😀xzz\nfive5\nfives\nsixsix\nsixsis\nwordy\nworldly\n\nzz\nz\n";

  static final Object[][] USER = {
    {"lucene", 9}, {"lucent", 2}, {"search", 7}, {"searches", 3}, {"apache", 5}, {"index", 8},
    {"indexes", 2}, {"spell", 4}, {"spelling", 1}, {"shell", 3}, {"checker", 2}, {"cafe", 2},
    {"naive", 1}, {"word", 3}, {"world", 6}, {"words", 1}, {"break", 2}, {"fast", 4},
    {"lucine", 1}, {"luceen", 1}, {"serach", 1}, {"aparche", 2}, {"wordly", 1}, {"fiver", 2},
  };

  static final String[] QUERIES = {
    "lucene", "lucenee", "lcuene", "lucen", "luc", "searh", "serach", "apahce", "indx", "indexx",
    "spel", "spelll", "chekcer", "cafe", "cafè", "naïv", "日本人", "wrold", "wordd", "brekfast",
    "abc", "abcde", "xyzzy", "search", "fivee", "sixsiz", "😀xyy", "a", "world", "wordl",
  };

  record Config(String name, Consumer<SpellChecker> setup, Float accuracyArg) {}

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("spell_checker_index");
    if (Files.exists(out)) IOUtils.rm(out);
    Path userDir = out.resolve("user");
    Files.createDirectories(userDir);

    int n = 20;
    List<StringBuilder> docs = new ArrayList<>();
    for (int d = 0; d < n; d++) docs.add(new StringBuilder());
    for (int i = 0; i < USER.length; i++) {
      for (int k = 0; k < (Integer) USER[i][1]; k++) {
        docs.get((i * 7 + k * 3) % n).append((String) USER[i][0]).append(' ');
      }
    }
    try (Directory dir = FSDirectory.open(userDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new WhitespaceAnalyzer());
      cfg.setUseCompoundFile(false);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (StringBuilder sb : docs) {
          Document doc = new Document();
          doc.add(new TextField("body", sb.toString(), Field.Store.NO));
          w.addDocument(doc);
        }
      }
    }

    List<Config> configs = new ArrayList<>();
    configs.add(new Config("default", s -> {}, null));
    configs.add(new Config("accuracy_0", s -> s.setAccuracy(0f), null));
    configs.add(new Config("accuracy_0.8", s -> s.setAccuracy(0.8f), null));
    configs.add(new Config("accuracy_arg_0.3", s -> {}, 0.3f));
    configs.add(new Config("frequency_comparator",
        s -> s.setComparator(new SuggestWordFrequencyComparator()), null));
    configs.add(new Config("distance_jarowinkler",
        s -> s.setStringDistance(new JaroWinklerDistance()), null));
    configs.add(new Config("distance_ngram", s -> s.setStringDistance(new NGramDistance()), null));
    configs.add(new Config("distance_lucene_levenshtein",
        s -> s.setStringDistance(new LuceneLevenshteinDistance()), null));

    StringBuilder suggest = new StringBuilder();
    StringBuilder exist = new StringBuilder();
    try (Directory user = FSDirectory.open(userDir);
        IndexReader ir = DirectoryReader.open(user)) {
      for (String stage : new String[] {"first", "second"}) {
        Path spellDir = out.resolve(stage);
        Files.createDirectories(spellDir);
        try (Directory dir = FSDirectory.open(spellDir)) {
          SpellChecker base = new SpellChecker(dir);
          base.indexDictionary(new PlainTextDictionary(new StringReader(DICTIONARY)),
              config(), true);
          if (stage.equals("second")) {
            base.indexDictionary(new LuceneDictionary(ir, "body"), config(), false);
          }
          for (String q : QUERIES) {
            exist.append(stage).append('\t').append(q).append('\t').append(base.exist(q)).append('\n');
          }
          base.close();
          for (Config c : configs) {
            SpellChecker s = new SpellChecker(dir);
            c.setup.accept(s);
            for (String q : QUERIES) {
              for (int num : new int[] {1, 3, 10}) {
                row(suggest, stage, c, q, num, "-",
                    c.accuracyArg == null
                        ? s.suggestSimilar(q, num)
                        : s.suggestSimilar(q, num, c.accuracyArg));
                for (SuggestMode mode : SuggestMode.values()) {
                  row(suggest, stage, c, q, num, mode.toString(),
                      c.accuracyArg == null
                          ? s.suggestSimilar(q, num, ir, "body", mode)
                          : s.suggestSimilar(q, num, ir, "body", mode, c.accuracyArg));
                }
              }
            }
            s.close();
          }
        }
      }
    }
    Files.writeString(out.resolve("suggest.tsv"), suggest.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("exist.tsv"), exist.toString(), StandardCharsets.UTF_8);
  }

  static IndexWriterConfig config() {
    IndexWriterConfig cfg = new IndexWriterConfig(null);
    cfg.setUseCompoundFile(false);
    return cfg;
  }

  static void row(StringBuilder sb, String stage, Config c, String q, int num, String mode,
      String[] got) {
    sb.append(stage).append('\t').append(c.name).append('\t').append(q).append('\t').append(num)
        .append('\t').append(mode);
    for (String w : got) sb.append('\t').append(w);
    sb.append('\n');
  }
}
