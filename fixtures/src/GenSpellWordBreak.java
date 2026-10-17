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
import org.apache.lucene.search.spell.CombineSuggestion;
import org.apache.lucene.search.spell.SuggestMode;
import org.apache.lucene.search.spell.SuggestWord;
import org.apache.lucene.search.spell.WordBreakSpellChecker;
import org.apache.lucene.search.spell.WordBreakSpellChecker.BreakSuggestionSortMethod;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.3e: {@code WordBreakSpellChecker} over a three-segment index. Writes {@code
 * spell_word_break/}: {@code index/}, {@code breaks.tsv} (configuration, query, mode, sort method,
 * maxSuggestions, then one column per suggestion: its words as {@code string:freq:scoreBits}
 * joined by spaces) and {@code combinations.tsv} (configuration, term list, mode, maxSuggestions,
 * then {@code indexes=string:freq:scoreBits} per suggestion).
 */
public class GenSpellWordBreak {
  static final Object[][] VOCABULARY = {
    {"good", 5}, {"morning", 3}, {"mor", 2}, {"ning", 4}, {"go", 6}, {"od", 1}, {"goo", 2},
    {"dmorning", 1}, {"the", 10}, {"quick", 4}, {"brown", 3}, {"fox", 2}, {"thequick", 1},
    {"hello", 4}, {"world", 3}, {"helloworld", 2}, {"hell", 2}, {"oworld", 1}, {"a", 8}, {"b", 7},
    {"c", 6}, {"ab", 3}, {"bc", 2}, {"abc", 1}, {"日本", 3}, {"語", 2}, {"😀", 2}, {"x", 3},
    {"😀x", 1}, {"aa", 4}, {"aaa", 2}, {"quickbrown", 2}, {"brownfox", 1}, {"goodmorning", 0},
  };

  static final String[] BREAK_QUERIES = {
    "goodmorning", "thequickbrown", "helloworld", "abc", "abcabc", "日本語", "😀x", "x",
    "goodmorningworld", "quickbrownfox", "zzz", "ab", "aaaa", "thequickbrownfox", "",
  };

  static final String[][] COMBINE_QUERIES = {
    {"good", "morning"},
    {"the", "quick", "brown", "fox"},
    {"hello", "world", "|", "x"},
    {"a", "b", "c"},
    {"日本", "語"},
    {"ab", "c", "a", "bc"},
    {"quick", "|", "brown", "fox"},
    {"a"},
    {"the", "quick", "zzz", "brown"},
    {"brown", "fox", "a", "a", "a"},
  };

  record Config(String name, Consumer<WordBreakSpellChecker> setup) {}

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("spell_word_break");
    Path indexDir = out.resolve("index");
    if (Files.exists(indexDir)) IOUtils.rm(indexDir);
    Files.createDirectories(indexDir);

    int n = 30;
    List<StringBuilder> docs = new ArrayList<>();
    for (int d = 0; d < n; d++) docs.add(new StringBuilder());
    for (int i = 0; i < VOCABULARY.length; i++) {
      String w = (String) VOCABULARY[i][0];
      int c = (Integer) VOCABULARY[i][1];
      for (int k = 0; k < c; k++) docs.get((i * 11 + k * 3) % n).append(w).append(' ');
    }
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new WhitespaceAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int d = 0; d < n; d++) {
          Document doc = new Document();
          doc.add(new StringField("id", Integer.toString(d), Field.Store.NO));
          doc.add(new TextField("wb", docs.get(d).toString(), Field.Store.NO));
          w.addDocument(doc);
          if (d % 10 == 9) w.commit();
        }
      }
    }

    List<Config> configs = new ArrayList<>();
    configs.add(new Config("default", s -> {}));
    configs.add(new Config("max_changes_2", s -> s.setMaxChanges(2)));
    configs.add(new Config("max_changes_3", s -> s.setMaxChanges(3)));
    configs.add(new Config("min_break_2", s -> {
      s.setMaxChanges(3);
      s.setMinBreakWordLength(2);
    }));
    configs.add(new Config("min_break_0", s -> s.setMinBreakWordLength(0)));
    configs.add(new Config("min_freq_2", s -> {
      s.setMaxChanges(2);
      s.setMinSuggestionFrequency(2);
    }));
    configs.add(new Config("max_evaluations_3", s -> {
      s.setMaxChanges(3);
      s.setMaxEvaluations(3);
    }));
    configs.add(new Config("max_evaluations_10", s -> {
      s.setMaxChanges(3);
      s.setMaxEvaluations(10);
    }));
    configs.add(new Config("max_combine_3", s -> {
      s.setMaxChanges(3);
      s.setMaxCombineWordLength(3);
    }));
    configs.add(new Config("max_combine_5", s -> {
      s.setMaxChanges(2);
      s.setMaxCombineWordLength(5);
    }));

    StringBuilder breaks = new StringBuilder();
    StringBuilder combos = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        IndexReader reader = DirectoryReader.open(dir)) {
      for (Config c : configs) {
        WordBreakSpellChecker s = new WordBreakSpellChecker();
        c.setup.accept(s);
        for (String q : BREAK_QUERIES) {
          for (SuggestMode mode : SuggestMode.values()) {
            for (BreakSuggestionSortMethod sort : BreakSuggestionSortMethod.values()) {
              for (int max : new int[] {0, 1, 2, 5, 20}) {
                SuggestWord[][] got = s.suggestWordBreaks(new Term("wb", q), max, reader, mode, sort);
                breaks.append(c.name).append('\t').append(q).append('\t').append(mode).append('\t')
                    .append(sort).append('\t').append(max);
                for (SuggestWord[] sugg : got) {
                  breaks.append('\t');
                  for (int i = 0; i < sugg.length; i++) {
                    if (i > 0) breaks.append(' ');
                    breaks.append(word(sugg[i]));
                  }
                }
                breaks.append('\n');
              }
            }
          }
        }
        for (String[] qs : COMBINE_QUERIES) {
          Term[] terms = new Term[qs.length];
          for (int i = 0; i < qs.length; i++) {
            terms[i] = qs[i].equals("|") ? WordBreakSpellChecker.SEPARATOR_TERM : new Term("wb", qs[i]);
          }
          for (SuggestMode mode : SuggestMode.values()) {
            for (int max : new int[] {0, 1, 3, 10}) {
              CombineSuggestion[] got = s.suggestWordCombinations(terms, max, reader, mode);
              combos.append(c.name).append('\t').append(String.join(" ", qs)).append('\t')
                  .append(mode).append('\t').append(max);
              for (CombineSuggestion cs : got) {
                combos.append('\t');
                int[] idx = cs.originalTermIndexes();
                for (int i = 0; i < idx.length; i++) {
                  if (i > 0) combos.append(',');
                  combos.append(idx[i]);
                }
                combos.append('=').append(word(cs.suggestion()));
              }
              combos.append('\n');
            }
          }
        }
      }
      // a null mode and sort method take the defaults
      WordBreakSpellChecker s = new WordBreakSpellChecker();
      SuggestWord[][] got = s.suggestWordBreaks(new Term("wb", "goodmorning"), 5, reader, null, null);
      breaks.append("nulls\tgoodmorning\tnull\tnull\t5");
      for (SuggestWord[] sugg : got) {
        breaks.append('\t');
        for (int i = 0; i < sugg.length; i++) {
          if (i > 0) breaks.append(' ');
          breaks.append(word(sugg[i]));
        }
      }
      breaks.append('\n');
    }
    Files.writeString(out.resolve("breaks.tsv"), breaks.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("combinations.tsv"), combos.toString(), StandardCharsets.UTF_8);
  }

  static String word(SuggestWord w) {
    return w.string + ":" + w.freq + ":" + Integer.toHexString(Float.floatToIntBits(w.score));
  }
}
